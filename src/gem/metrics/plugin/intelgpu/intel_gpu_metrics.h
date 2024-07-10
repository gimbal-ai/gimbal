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

#include <absl/container/flat_hash_map.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/event/dispatcher.h"
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
  IntelGPUMetrics(::gml::metrics::MetricsSystem* metrics_system, gml::event::Dispatcher*);

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
