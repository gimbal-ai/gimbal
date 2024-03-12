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
#include <chrono>

#include <absl/container/flat_hash_map.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/system/proc_parser.h"
#include "src/gem/metrics/core/scraper_builder.h"

namespace gml::gem::metrics::intelgpu {

struct DeviceMetrics {
  uint64_t system_counter_ns;
  uint64_t gem_counter_ns;
};

class IntelGPUMetrics : public core::Scraper {
 public:
  IntelGPUMetrics() = delete;
  explicit IntelGPUMetrics(::gml::metrics::MetricsSystem* metrics_system);

  void Scrape() override;

 private:
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> system_utilization_counter_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> system_memory_size_gauge_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> gem_utilization_counter_;

  absl::base_internal::SpinLock metrics_lock_;
  absl::flat_hash_map<std::string, DeviceMetrics> device_metrics_ ABSL_GUARDED_BY(metrics_lock_);
  std::chrono::steady_clock::time_point start_time_;

  gml::system::ProcParser proc_parser_;

  Status ScrapeWithError();
};

}  // namespace gml::gem::metrics::intelgpu
