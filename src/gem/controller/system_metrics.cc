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

#include "src/gem/controller/system_metrics.h"
#include "src/common/base/utils.h"
#include "src/common/system/config.h"
#include "src/common/system/proc_parser.h"
#include "src/common/system/proc_pid_path.h"

namespace gml::gem::controller {

SystemMetricsReader::SystemMetricsReader(::gml::metrics::MetricsSystem* metrics_system)
    : metrics::Scrapeable(metrics_system) {
  CHECK(metrics_system != nullptr);
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");
  cpu_stats_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("system.cpu.time"));
  cpu_num_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("system.cpu.virtual.count"));
  mem_stats_total_bytes_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.memory.total_bytes"));
  mem_stats_free_bytes_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.memory.free_bytes"));
  network_rx_bytes_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.network.rx_bytes"));
  network_rx_drops_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.network.rx_drops"));
  network_tx_bytes_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.network.tx_bytes"));
  network_tx_drops_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("system.network.tx_drops"));
}

void SystemMetricsReader::Scrape() {
  // Add CPU metrics for system.
  std::vector<gml::system::ProcParser::CPUStats> stats;
  GML_CHECK_OK(proc_parser_.ParseProcStatAllCPUs(&stats));

  for (const auto& [c, stat] : Enumerate(stats)) {
    auto cpu = std::to_string(c);
    cpu_stats_counter_->Add(stat.cpu_ktime_ns, {{"cpu", cpu}, {"state", "system"}}, {});
    cpu_stats_counter_->Add(stat.cpu_utime_ns, {{"cpu", cpu}, {"state", "user"}}, {});
    cpu_stats_counter_->Add(stat.cpu_idletime_ns, {{"cpu", cpu}, {"state", "idle"}}, {});
    cpu_stats_counter_->Add(stat.cpu_iowaittime_ns, {{"cpu", cpu}, {"state", "wait"}}, {});
  }

  // Add memory metrics for system.
  cpu_num_counter_->Add(static_cast<int64_t>(stats.size()), {{"state", "system"}}, {});
  gml::system::ProcParser::SystemStats system_stats;
  GML_CHECK_OK(proc_parser_.ParseProcStat(&system_stats));
  mem_stats_total_bytes_->Add(system_stats.mem_total_bytes, {{"state", "system"}}, {});
  mem_stats_free_bytes_->Add(system_stats.mem_free_bytes, {{"state", "system"}}, {});

  std::vector<gml::system::ProcParser::NetworkStats> network_stats;
  auto s = proc_parser_.ParseProcNetDev(&network_stats);
  if (!s.ok()) {
    LOG(INFO) << "Failed to read proc network stats. Skipping...";
    return;
  }

  for (auto n : network_stats) {
    network_rx_bytes_counter_->Add(n.rx_bytes, {{"interface", n.interface}}, {});
    network_rx_drops_counter_->Add(n.rx_drops, {{"interface", n.interface}}, {});
    network_tx_bytes_counter_->Add(n.tx_bytes, {{"interface", n.interface}}, {});
    network_tx_drops_counter_->Add(n.tx_drops, {{"interface", n.interface}}, {});
  }
}
}  // namespace gml::gem::controller
