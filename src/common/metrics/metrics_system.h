/*
 * Copyright 2023- Gimlet Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <opentelemetry/exporters/otlp/otlp_metric_utils.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/sdk/metrics/meter.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/metric_reader.h>

#include "src/common/base/base.h"

namespace gml::metrics {

// Bounds are based on the default histogram suggestions in the Otel Spec.
// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/sdk.md#explicit-bucket-histogram-aggregation
static const std::vector<double> kDefaultHistogramBounds = {
    0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};

class MetricsSystem;

class Scrapeable {
 public:
  Scrapeable() = delete;
  /*
   * Scrapeable object with metrics_system. The lifetime of metrics_system needs to outlive the
   * scrapeable.
   */
  explicit Scrapeable(MetricsSystem*);
  virtual ~Scrapeable();

  /**
   * Scrapeables need to define a Scrape function that will be called before new metrics will be
   * pulled.
   */
  virtual void Scrape() = 0;

 protected:
  MetricsSystem* metrics_system_;
};

class AuxMetricsProvider {
 public:
  virtual ~AuxMetricsProvider();

  /**
   * Needs to be defined by implementer. Must populate stats.
   */
  virtual Status CollectMetrics(opentelemetry::proto::metrics::v1::ResourceMetrics* metrics) = 0;
};

/**
 * MetricsSystem is a wrapper around OTel metrics. It is structured such that it is
 * system-wide, with a single instance for the whole system.
 *
 * Creating stats:
 *
 * GetMeterProvider() can be used to create and add to new stats.
 * This pretty much gives direct access to OTel, so please follow OTel documentation for types
 * of stats and how to use them. There is a basic example included in the test for this file.
 *
 * Consuming stats:
 *
 * On the reader side, CollectAllAsProto() can be used to extract all the stats in an OTel
 * defined protobuf, or Reader() can be called to set-up your own callback of how to process the
 * stats.
 *
 * The initial implementation of this metrics system is intentionally simple, avoid things like
 * OTel views. We may add these in the future, depending on the complexity of the stats collection.
 */
class MetricsSystem {
 public:
  /**
   * Get the instance to either add/update stats, or read stats out.
   */
  static MetricsSystem& GetInstance();

  /**
   * Main interface to creating and updating statistics. See OTel or test code for examples
   * of how to use it.
   */
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider> GetMeterProvider();

  /**
   * Returns all the stats in the system.
   */
  opentelemetry::proto::metrics::v1::ResourceMetrics CollectAllAsProto();

  std::vector<opentelemetry::proto::metrics::v1::ResourceMetrics> ChunkMetrics(
      opentelemetry::proto::metrics::v1::ResourceMetrics* metrics, uint64_t chunk_size);

  /**
   * Returns the reader for custom access to the Collect() call.
   */
  opentelemetry::sdk::metrics::MetricReader* Reader();

  Status RegisterScraper(Scrapeable* s);

  const absl::flat_hash_set<Scrapeable*>& scrapeables() const { return scrapeables_; };

  Status UnRegisterScraper(Scrapeable* s);

  Status RegisterAuxMetricsProvider(AuxMetricsProvider* p);
  void UnRegisterAuxMetricsProvider(AuxMetricsProvider* p);
  const absl::flat_hash_set<AuxMetricsProvider*>& aux_metrics_providers() const {
    return aux_metrics_providers_;
  }

  /**
   * Returns a histogram with the given custom bounds.
   */
  template <typename T>
  opentelemetry::metrics::Histogram<T>* GetOrCreateHistogramWithBounds(
      std::string name, const std::string& description, const std::vector<double>& bounds) {
    static_assert(std::is_same_v<T, uint64_t> || std::is_same_v<T, double>,
                  "Only uint64_t or double supported for GetOrCreateHistogramWithBounds");
    using HistogramUPtr = std::unique_ptr<opentelemetry::metrics::Histogram<T>>;

    absl::base_internal::SpinLockHolder lock(&meter_provider_lock_);

    if (metrics_.contains(name)) {
      if (!std::holds_alternative<HistogramUPtr>(metrics_[name])) {
        LOG(FATAL) << absl::Substitute("Metric $0 is not a histogram of the appropriate type",
                                       name);
      }
      return std::get<HistogramUPtr>(metrics_[name]).get();
    }

    auto provider = GetMeterProvider();
    auto meter = provider->GetMeter("gml");

    auto instrument_selector = std::make_unique<opentelemetry::sdk::metrics::InstrumentSelector>(
        opentelemetry::sdk::metrics::InstrumentType::kHistogram, name, "");
    auto* sdk_meter = static_cast<opentelemetry::sdk::metrics::Meter*>(meter.get());
    auto* scope = sdk_meter->GetInstrumentationScope();
    auto meter_selector = std::make_unique<opentelemetry::sdk::metrics::MeterSelector>(
        scope->GetName(), scope->GetVersion(), scope->GetSchemaURL());

    auto histogram_aggregation_config =
        std::make_shared<opentelemetry::sdk::metrics::HistogramAggregationConfig>();
    histogram_aggregation_config->boundaries_ = bounds;
    auto aggregation_config = std::shared_ptr<opentelemetry::sdk::metrics::AggregationConfig>(
        std::move(histogram_aggregation_config));
    auto histogram_view = std::make_unique<opentelemetry::sdk::metrics::View>(
        name, "", "", opentelemetry::sdk::metrics::AggregationType::kHistogram,
        std::move(aggregation_config));

    auto* sdk_provider = static_cast<opentelemetry::sdk::metrics::MeterProvider*>(provider.get());
    sdk_provider->AddView(std::move(instrument_selector), std::move(meter_selector),
                          std::move(histogram_view));

    HistogramUPtr histogram;
    if constexpr (std::is_same_v<T, uint64_t>) {
      histogram = meter->CreateUInt64Histogram(name, description);
    } else if constexpr (std::is_same_v<T, double>) {
      histogram = meter->CreateDoubleHistogram(name, description);
    }

    // Store the histogram in the map, so we can return it if it is requested again.
    auto histogram_ptr = histogram.get();
    metrics_[name] = std::move(histogram);
    return histogram_ptr;
  }

  opentelemetry::metrics::Counter<uint64_t>* GetOrCreateCounter(const std::string& name,
                                                                const std::string& description) {
    using CounterUPtr = std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>>;

    absl::base_internal::SpinLockHolder lock(&meter_provider_lock_);

    if (metrics_.contains(name)) {
      if (!std::holds_alternative<CounterUPtr>(metrics_[name])) {
        LOG(FATAL) << absl::Substitute("Metric $0 is not a counter", name);
      }
      return std::get<CounterUPtr>(metrics_[name]).get();
    }

    auto provider = GetMeterProvider();
    auto meter = provider->GetMeter("gml");

    CounterUPtr counter = meter->CreateUInt64Counter(name, description);

    // Store the counter in the map, so we can return it if it is requested again.
    auto counter_ptr = counter.get();
    metrics_[name] = std::move(counter);
    return counter_ptr;
  }

  template <typename T>
  opentelemetry::metrics::Gauge<T>* GetOrCreateGauge(const std::string& name,
                                                     const std::string& description) {
    using GaugeUPtr = std::unique_ptr<opentelemetry::metrics::Gauge<T>>;

    static_assert(std::is_same_v<T, uint64_t> || std::is_same_v<T, double>,
                  "Only uint64_t or double supported for GetOrCreateGauge");

    absl::base_internal::SpinLockHolder lock(&meter_provider_lock_);

    if (metrics_.contains(name)) {
      if (!std::holds_alternative<GaugeUPtr>(metrics_[name])) {
        LOG(FATAL) << absl::Substitute("Metric $0 is not a gauge of the appropriate type", name);
      }
      return std::get<GaugeUPtr>(metrics_[name]).get();
    }

    auto provider = GetMeterProvider();
    auto meter = provider->GetMeter("gml");

    GaugeUPtr gauge;
    if constexpr (std::is_same_v<T, uint64_t>) {
      gauge = meter->CreateInt64Gauge(name, description);
    } else if constexpr (std::is_same_v<T, double>) {
      gauge = meter->CreateDoubleGauge(name, description);
    }

    // Store the gauge in the map, so we can return it if it is requested again.
    auto gauge_ptr = gauge.get();
    metrics_[name] = std::move(gauge);
    return gauge_ptr;
  }

  // Reset removes all metrics.
  void Reset();

 private:
  MetricsSystem() { Init(); }

  void Init();

  absl::base_internal::SpinLock scrapeables_lock_;
  absl::flat_hash_set<Scrapeable*> scrapeables_ ABSL_GUARDED_BY(scrapeables_lock_);

  absl::base_internal::SpinLock aux_metrics_providers_lock_;
  absl::flat_hash_set<AuxMetricsProvider*> aux_metrics_providers_
      ABSL_GUARDED_BY(aux_metrics_providers_lock_);

  // MeterProvider::AddMetricReader and MeterProvider::AddView are not thread safe, so we need to
  // lock for those method calls.
  // TODO(james): Currently, we allow anyone to call GetMeterProvider and then call these unsafe
  // APIs, we should refactor this to ensure calls to these methods are only possible from within
  // metrics system.
  absl::base_internal::SpinLock meter_provider_lock_;
  std::shared_ptr<opentelemetry::sdk::metrics::MetricReader> reader_;

  // Maps to all created metrics. If a metric is requested again, we return the same instance.

  using MetricInstrumentsPointers =
      std::variant<std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>>,
                   std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>>,
                   std::unique_ptr<opentelemetry::metrics::Gauge<double>>,
                   std::unique_ptr<opentelemetry::metrics::Histogram<double>>,
                   std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>>>;
  absl::flat_hash_map<std::string, MetricInstrumentsPointers> metrics_;
};

}  // namespace gml::metrics
