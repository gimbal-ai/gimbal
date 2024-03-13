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
#include <fstream>

#include <opentelemetry/metrics/sync_instruments.h>

#include "src/common/event/dispatcher.h"
#include "src/common/event/timer.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/system/proc_parser.h"
#include "src/gem/metrics/core/scraper_builder.h"

namespace gml::gem::metrics::jetson {
class JetsonGPUMetrics : public core::Scraper {
 public:
  ~JetsonGPUMetrics() override = default;
  JetsonGPUMetrics() = delete;
  JetsonGPUMetrics(gml::metrics::MetricsSystem* metrics_system, gml::event::Dispatcher* dispatcher);

  void Scrape() override;

 private:
  gml::event::Dispatcher* dispatcher_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> system_memory_size_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> system_memory_usage_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> gem_memory_usage_gauge_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> system_utilization_counter_;

  system::ProcParser proc_parser_;

  gml::event::TimerUPtr util_collect_timer_;
  absl::base_internal::SpinLock utilization_lock_;
  uint64_t system_utilization_ns_ ABSL_GUARDED_BY(utilization_lock_) = 0;
  std::ifstream load_file_;
  std::string line_buf_;
  std::chrono::high_resolution_clock::time_point prev_collect_;

  Status ScrapeWithError();
  Status InitUtilizationCollection();
  void CollectUtilization();
};
}  // namespace gml::gem::metrics::jetson
