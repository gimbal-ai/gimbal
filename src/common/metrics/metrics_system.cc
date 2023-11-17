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

#include "src/common/metrics/metrics_system.h"

#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/metric_reader.h"

using opentelemetry::sdk::metrics::AggregationTemporality;
using opentelemetry::sdk::metrics::InstrumentType;

namespace gml::metrics {

namespace {
std::unique_ptr<MetricsSystem> g_instance;
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
  if (g_instance != nullptr) {
    return;
  }

  std::shared_ptr<opentelemetry::metrics::MeterProvider> provider(
      new opentelemetry::sdk::metrics::MeterProvider());
  std::shared_ptr<opentelemetry::sdk::metrics::MeterProvider> p =
      std::static_pointer_cast<opentelemetry::sdk::metrics::MeterProvider>(provider);

  auto reader = std::shared_ptr<opentelemetry::sdk::metrics::MetricReader>(new BasicMetricReader());
  p->AddMetricReader(reader);

  opentelemetry::metrics::Provider::SetMeterProvider(provider);

  g_instance = std::unique_ptr<MetricsSystem>(new MetricsSystem());
  g_instance->reader_ = std::move(reader);
}

void MetricsSystem::ResetInstance() {
  g_instance.reset();
  Init();
}

MetricsSystem& MetricsSystem::GetInstance() {
  if (g_instance == nullptr) {
    MetricsSystem::ResetInstance();
  };
  return *g_instance;
}

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider>
MetricsSystem::GetMeterProvider() {
  auto p = opentelemetry::metrics::Provider::GetMeterProvider();
  CHECK(p != nullptr);
  return p;
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

}  // namespace gml::metrics
