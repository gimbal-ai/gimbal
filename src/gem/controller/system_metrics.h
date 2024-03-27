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
#include "src/common/system/cpu_info_reader.h"
#include "src/common/system/proc_parser.h"

namespace gml::gem::controller {

class SystemMetricsReader : public gml::metrics::Scrapeable {
 public:
  SystemMetricsReader() = delete;
  explicit SystemMetricsReader(::gml::metrics::MetricsSystem* metrics_system,
                               std::unique_ptr<gml::system::CPUInfoReader> cpu_reader);
  ~SystemMetricsReader() override = default;
  void Scrape() override;

  template <typename T>
  void GetObservableResultFromNetworkStats(
      opentelemetry::metrics::ObserverResult observer,
      const std::function<int64_t(const gml::system::ProcParser::NetworkStats& network_stats)>&
          get_stat);

 private:
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> cpu_stats_counter_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> cpu_num_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> cpu_frequency_gauge_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_stats_total_bytes_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_stats_free_bytes_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_stats_buffered_bytes_;
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> mem_stats_cached_bytes_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_rx_bytes_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_rx_drops_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_tx_bytes_counter_;
  std::shared_ptr<opentelemetry::metrics::ObservableInstrument> network_tx_drops_counter_;

  gml::system::ProcParser proc_parser_;
  std::unique_ptr<gml::system::CPUInfoReader> cpu_info_reader_;

  std::vector<gml::system::ProcParser::CPUStats> cpu_stats_;
  std::vector<gml::system::ProcParser::NetworkStats> network_stats_;
};

}  // namespace gml::gem::controller
