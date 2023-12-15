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
  ;
  absl::flat_hash_map<pid_t, std::vector<gml::system::ProcParser::NetworkStats>> pid_network_stats_
      ABSL_GUARDED_BY(pid_stats_lock_);
};

}  // namespace gml::gem::controller
