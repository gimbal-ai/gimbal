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
