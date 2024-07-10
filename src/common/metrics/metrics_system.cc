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

#include "src/common/metrics/metrics_system.h"

#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/metric_reader.h>

using opentelemetry::sdk::metrics::AggregationTemporality;
using opentelemetry::sdk::metrics::InstrumentType;

namespace gml::metrics {

Scrapeable::Scrapeable(MetricsSystem* metrics_system) : metrics_system_(metrics_system) {
  GML_CHECK_OK(metrics_system_->RegisterScraper(this));
}

Scrapeable::~Scrapeable() { GML_CHECK_OK(metrics_system_->UnRegisterScraper(this)); }

AuxMetricsProvider::~AuxMetricsProvider() {
  MetricsSystem::GetInstance().UnRegisterAuxMetricsProvider(this);
}

/**
 * A basic metric reader. In our pull-based model, we really want access to the Collect() call,
 * which is defined in the base class. But we must still define a MetricReader to initialize Otel.
 */
class BasicMetricReader : public opentelemetry::sdk::metrics::MetricReader {
 public:
  BasicMetricReader() = default;

  AggregationTemporality GetAggregationTemporality(
      InstrumentType /* instrument_type */) const noexcept override {
    return AggregationTemporality::kCumulative;
  }

  bool OnShutDown(std::chrono::microseconds /* timeout */) noexcept override { return true; }
  bool OnForceFlush(std::chrono::microseconds /* timeout */) noexcept override { return true; }
  void OnInitialized() noexcept override {}
};

// Sets up the Metrics system. Only needs to be called once.
// Called automatically by GetInstance() on first access.
void MetricsSystem::Init() {
  absl::base_internal::SpinLockHolder scrape_lock(&scrapeables_lock_);
  absl::base_internal::SpinLockHolder mp_lock(&meter_provider_lock_);

  scrapeables_.clear();
  reader_.reset();

  std::shared_ptr<opentelemetry::metrics::MeterProvider> provider(
      new opentelemetry::sdk::metrics::MeterProvider());
  std::shared_ptr<opentelemetry::sdk::metrics::MeterProvider> p =
      std::static_pointer_cast<opentelemetry::sdk::metrics::MeterProvider>(provider);

  auto reader = std::shared_ptr<opentelemetry::sdk::metrics::MetricReader>(new BasicMetricReader());
  p->AddMetricReader(reader);

  opentelemetry::metrics::Provider::SetMeterProvider(provider);
  reader_ = std::move(reader);
}

void MetricsSystem::Reset() { Init(); }

MetricsSystem& MetricsSystem::GetInstance() {
  static MetricsSystem metrics_system;
  return metrics_system;
}

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider>
MetricsSystem::GetMeterProvider() {
  auto p = opentelemetry::metrics::Provider::GetMeterProvider();
  CHECK(p != nullptr);
  return p;
}

std::vector<opentelemetry::proto::metrics::v1::ResourceMetrics> MetricsSystem::ChunkMetrics(
    opentelemetry::proto::metrics::v1::ResourceMetrics* metrics, uint64_t chunk_size) {
  std::vector<opentelemetry::proto::metrics::v1::ResourceMetrics> chunks;
  for (const auto& scope_metric : metrics->scope_metrics()) {
    opentelemetry::proto::metrics::v1::ResourceMetrics chunk;

    // Create new scope metric, copying over resource/scope info.
    *chunk.mutable_resource() = metrics->resource();
    auto scope_metric_chunk = chunk.add_scope_metrics();
    *scope_metric_chunk->mutable_scope() = scope_metric.scope();

    for (const auto& metric : scope_metric.metrics()) {
      if (chunk.ByteSizeLong() + metric.ByteSizeLong() > chunk_size) {
        chunks.push_back(chunk);

        // Create new scope metric, now that we're starting a new chunk.
        chunk.Clear();
        *chunk.mutable_resource() = metrics->resource();
        scope_metric_chunk = chunk.add_scope_metrics();
        *scope_metric_chunk->mutable_scope() = scope_metric.scope();
      }
      *scope_metric_chunk->add_metrics() = metric;
    }

    if (chunk.scope_metrics(0).metrics_size() > 0) {
      chunks.push_back(chunk);
    }
  }

  return chunks;
}

opentelemetry::proto::metrics::v1::ResourceMetrics MetricsSystem::CollectAllAsProto() {
  opentelemetry::proto::metrics::v1::ResourceMetrics metrics;

  auto results_cb = [&metrics](opentelemetry::sdk::metrics::ResourceMetrics& metric_data) {
    opentelemetry::exporter::otlp::OtlpMetricUtils::PopulateResourceMetrics(metric_data, &metrics);
    return true;
  };

  reader_->Collect(results_cb);

  return metrics;
}

opentelemetry::sdk::metrics::MetricReader* MetricsSystem::Reader() { return reader_.get(); };

Status MetricsSystem::RegisterScraper(Scrapeable* s) {
  absl::base_internal::SpinLockHolder lock(&scrapeables_lock_);
  // Check and make sure it hasn't already been registered.
  auto [_, inserted] = scrapeables_.emplace(s);
  if (!inserted) {
    return error::AlreadyExists("scraper has already been registered");
  }
  return Status::OK();
}

Status MetricsSystem::UnRegisterScraper(Scrapeable* s) {
  absl::base_internal::SpinLockHolder lock(&scrapeables_lock_);
  bool erased = scrapeables_.erase(s);
  if (!erased) {
    return error::NotFound("scraper not found");
  }
  return Status::OK();
}

Status MetricsSystem::RegisterAuxMetricsProvider(AuxMetricsProvider* p) {
  absl::base_internal::SpinLockHolder lock(&aux_metrics_providers_lock_);
  auto [_, inserted] = aux_metrics_providers_.emplace(p);
  if (!inserted) {
    return error::AlreadyExists("auxiliary stats provider has already been registered.");
  }
  return Status::OK();
}

void MetricsSystem::UnRegisterAuxMetricsProvider(AuxMetricsProvider* p) {
  absl::base_internal::SpinLockHolder lock(&aux_metrics_providers_lock_);
  aux_metrics_providers_.erase(p);
}

}  // namespace gml::metrics
