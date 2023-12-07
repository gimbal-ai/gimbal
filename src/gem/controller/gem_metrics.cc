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

#include <unistd.h>

#include "src/common/base/utils.h"
#include "src/common/system/config.h"
#include "src/common/system/proc_parser.h"
#include "src/common/system/proc_pid_path.h"
#include "src/gem/controller/gem_metrics.h"

namespace gml::gem::controller {

GEMMetricsReader::GEMMetricsReader(::gml::metrics::MetricsSystem* metrics_system)
    : metrics::Scrapeable(metrics_system) {
  CHECK(metrics_system != nullptr);
  pgid_ = getpgid(getpid());

  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");
  cpu_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.cpu.time"));
  mem_usage_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.memory.usage"));
  mem_virtual_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.memory.virtual"));
  thread_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.threads"));
  context_switches_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.context_switches"));
  network_rx_bytes_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.network.rx_bytes"));
  network_rx_drops_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.network.rx_drops"));
  network_tx_bytes_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.network.tx_bytes"));
  network_tx_drops_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.network.tx_drops"));
  disk_rchar_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.disk.rchar"));
  disk_wchar_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.disk.wchar"));
  disk_read_bytes_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("gem.disk.read_bytes"));
  disk_write_bytes_counter_ =
      std::move(gml_meter->CreateInt64UpDownCounter("gem.disk.write_bytes"));
}

void GEMMetricsReader::Scrape() {
  // Generate metrics for all child processes in parent.
  auto pids = proc_parser_.ListChildPIDsForPGID(getpgid(getpid()));
  for (auto p : pids) {
    gml::system::ProcParser::ProcessStats process_stats;
    auto s = proc_parser_.ParseProcPIDStat(p, gml::system::Config::GetInstance().PageSizeBytes(),
                                           gml::system::Config::GetInstance().KernelTickTimeNS(),
                                           &process_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc stats. Skipping...";
      continue;
    }

    auto pid = std::to_string(p);

    cpu_counter_->Add(process_stats.ktime_ns, {{"pid", pid}, {"state", "user"}}, {});
    cpu_counter_->Add(process_stats.utime_ns, {{"pid", pid}, {"state", "system"}}, {});
    mem_usage_counter_->Add(process_stats.rss_bytes, {{"pid", pid}, {"state", "system"}}, {});
    mem_virtual_counter_->Add(static_cast<int64_t>(process_stats.vsize_bytes),
                              {{"pid", pid}, {"state", "system"}}, {});
    thread_counter_->Add(process_stats.num_threads, {{"pid", pid}, {"state", "system"}}, {});
    gml::system::ProcParser::ProcessStatus process_status;
    s = proc_parser_.ParseProcPIDStatus(p, &process_status);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc status. Skipping...";
      continue;
    }

    context_switches_counter_->Add(
        process_status.voluntary_ctxt_switches,
        {{"pid", pid}, {"state", "system"}, {"context_switch_type", "voluntary"}}, {});
    context_switches_counter_->Add(
        process_status.nonvoluntary_ctxt_switches,
        {{"pid", pid}, {"state", "system"}, {"context_switch_type", "involuntary"}}, {});

    std::vector<gml::system::ProcParser::NetworkStats> network_stats;
    s = proc_parser_.ParseProcPIDNetDev(p, &network_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc network stats. Skipping...";
      continue;
    }

    for (auto n : network_stats) {
      network_rx_bytes_counter_->Add(n.rx_bytes, {{"interface", n.interface}, {"pid", pid}}, {});
      network_rx_drops_counter_->Add(n.rx_drops, {{"interface", n.interface}, {"pid", pid}}, {});
      network_tx_bytes_counter_->Add(n.tx_bytes, {{"interface", n.interface}, {"pid", pid}}, {});
      network_tx_drops_counter_->Add(n.tx_drops, {{"interface", n.interface}, {"pid", pid}}, {});
    }

    s = proc_parser_.ParseProcPIDStatIO(p, &process_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc IO stats. Skipping...";
      continue;
    }

    disk_rchar_counter_->Add(process_stats.rchar_bytes, {{"pid", pid}}, {});
    disk_wchar_counter_->Add(process_stats.wchar_bytes, {{"pid", pid}}, {});
    disk_read_bytes_counter_->Add(process_stats.read_bytes, {{"pid", pid}}, {});
    disk_write_bytes_counter_->Add(process_stats.write_bytes, {{"pid", pid}}, {});
  }
}

}  // namespace gml::gem::controller
