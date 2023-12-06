/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
 */

#pragma once

#include <opentelemetry/exporters/otlp/otlp_metric_utils.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/sdk/metrics/meter.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/metric_reader.h>

#include "src/common/base/base.h"

namespace gml::metrics {

class MetricsSystem;

class Scrapeable {
 public:
  Scrapeable() = delete;
  /*
   * Scrapeable object with metrics_system. The lifetime of metrics_system needs to be outlive the
   * scrapeable.
   */
  explicit Scrapeable(MetricsSystem*);
  virtual ~Scrapeable();

  /**
   * Scrapeables need to define a Scrape function that wil be called before new metrics will be
   * pulled.
   */
  virtual void Scrape() = 0;

 protected:
  MetricsSystem* metrics_system_;
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
 * On the reader side, the CollectAllAsProto() can be used to extract all the stats in an OTel
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

  static void ResetInstance();

  /**
   * Main interface to creating and updating statistics. See OTel or test code for examples
   * of how to use it.
   */
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider> GetMeterProvider();

  /**
   * Returns all the stats in the system.
   */
  opentelemetry::proto::metrics::v1::ResourceMetrics CollectAllAsProto();

  /**
   * Returns the reader for custom access to the Collect() call.
   */
  opentelemetry::sdk::metrics::MetricReader* Reader();

  Status RegisterScraper(Scrapeable* s) {
    absl::base_internal::SpinLockHolder lock(&scrapeables_lock_);
    // Check and make sure it hasn't already been registered.
    if (std::find(scrapeables_.begin(), scrapeables_.end(), s) != scrapeables_.end()) {
      return error::AlreadyExists("scraper has already been registered");
    }
    scrapeables_.emplace_back(s);
    return Status::OK();
  }

  const std::vector<Scrapeable*>& scrapeables() const { return scrapeables_; };

  Status UnRegisterScraper(Scrapeable* s) {
    absl::base_internal::SpinLockHolder lock(&scrapeables_lock_);
    auto it = std::find(scrapeables_.begin(), scrapeables_.end(), s);
    if (it == scrapeables_.end()) {
      return error::NotFound("scraper not found");
    }
    scrapeables_.erase(it);
    return Status::OK();
  }

  /**
   * Returns a histogram with the given custom bounds.
   */
  template <typename T>
  std::unique_ptr<opentelemetry::metrics::Histogram<T>> CreateHistogramWithBounds(
      std::string name, const std::vector<double>& bounds) {
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

    std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
    if constexpr (std::is_same_v<T, uint64_t>) {
      histogram = meter->CreateUInt64Histogram(name);
    } else if constexpr (std::is_same_v<T, double>) {
      histogram = meter->CreateDoubleHistogram(name);
    } else {
      CHECK(false) << "Only uint64_t or double supported for CreateHistogramWithBounds";
    }
    return histogram;
  }

 private:
  MetricsSystem() = default;

  static void Init();

  absl::base_internal::SpinLock scrapeables_lock_;
  std::vector<Scrapeable*> scrapeables_ ABSL_GUARDED_BY(scrapeables_lock_);
  std::shared_ptr<opentelemetry::sdk::metrics::MetricReader> reader_;
};

}  // namespace gml::metrics
