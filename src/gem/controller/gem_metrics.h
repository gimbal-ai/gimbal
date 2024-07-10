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
#include <opentelemetry/metrics/provider.h>

#include "src/common/metrics/metrics_system.h"
#include "src/common/system/proc_parser.h"

namespace gml::gem::controller {

class GEMMetricsReader : public gml::metrics::Scrapeable {
 public:
  GEMMetricsReader() = delete;
  explicit GEMMetricsReader(::gml::metrics::MetricsSystem* metrics_system);
  ~GEMMetricsReader() override = default;
  void Scrape() override;

  template <typename T>
  void GetObservableResultFromNetworkStats(
      opentelemetry::metrics::ObserverResult observer,
      const std::function<int64_t(const system::ProcParser::NetworkStats& process_stats)>&
          get_stat);
  template <typename T>
  void GetObservableResultFromProcessStats(
      opentelemetry::metrics::ObserverResult observer,
      const std::function<int64_t(const system::ProcParser::ProcessStats& network_stats)>&
          get_stat);

 private:
  pid_t pid_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> cpu_counter_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_usage_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_virtual_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> thread_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> privileged_gauge_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> context_switches_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_rx_bytes_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_rx_drops_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_tx_bytes_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_tx_drops_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> disk_rchar_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> disk_wchar_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> disk_read_bytes_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> disk_write_bytes_counter_;

  gml::system::ProcParser proc_parser_;

  absl::base_internal::SpinLock pid_stats_lock_;
  absl::flat_hash_map<pid_t, gml::system::ProcParser::ProcessStats> pid_process_stats_
      ABSL_GUARDED_BY(pid_stats_lock_);
  absl::flat_hash_map<pid_t, gml::system::ProcParser::ProcessStatus> pid_process_status_
      ABSL_GUARDED_BY(pid_stats_lock_);
  std::vector<gml::system::ProcParser::NetworkStats> pid_network_stats_;
  absl::flat_hash_map<pid_t, pid_t> pid_to_tgid_ ABSL_GUARDED_BY(pid_stats_lock_);
};

}  // namespace gml::gem::controller
