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

namespace {
template <typename T>
auto GetObservableResult(opentelemetry::metrics::ObserverResult& observer) {
  return std::get<std::shared_ptr<opentelemetry::metrics::ObserverResultT<T>>>(observer);
}
}  // namespace

SystemMetricsReader::SystemMetricsReader(::gml::metrics::MetricsSystem* metrics_system,
                                         std::unique_ptr<gml::system::CPUInfoReader> cpu_reader)
    : metrics::Scrapeable(metrics_system) {
  CHECK(metrics_system != nullptr);
  cpu_info_reader_ = std::move(cpu_reader);

  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  cpu_stats_counter_ = gml_meter->CreateDoubleObservableCounter("gml.system.cpu.seconds.total");
  cpu_stats_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<SystemMetricsReader*>(parent);
        for (const auto& [c, stat] : Enumerate(reader->cpu_stats_)) {
          auto cpu = std::to_string(c);
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(stat.cpu_ktime_ns) / 1E9, {{"cpu", cpu}, {"state", "system"}});
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(stat.cpu_utime_ns) / 1E9, {{"cpu", cpu}, {"state", "user"}});
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(stat.cpu_idletime_ns) / 1E9, {{"cpu", cpu}, {"state", "idle"}});
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(stat.cpu_iowaittime_ns) / 1E9, {{"cpu", cpu}, {"state", "wait"}});
        }
      },
      this);

  cpu_num_gauge_ = gml_meter->CreateInt64Gauge("gml.system.cpu.virtual");
  cpu_frequency_gauge_ = gml_meter->CreateInt64Gauge("gml.system.cpu.scaling_frequency_hertz");
  mem_stats_total_bytes_ = gml_meter->CreateInt64Gauge("gml.system.memory.total_bytes");
  mem_stats_free_bytes_ = gml_meter->CreateInt64Gauge("gml.system.memory.free_bytes");
  mem_stats_buffered_bytes_ = gml_meter->CreateInt64Gauge("gml.system.memory.buffered_bytes");
  mem_stats_cached_bytes_ = gml_meter->CreateInt64Gauge("gml.system.memory.cached_bytes");

  // Setup network counters.
  network_rx_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.system.network.rx_bytes.total");
  network_rx_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<SystemMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(observer,
                                                             [](auto p) { return p.rx_bytes; });
      },
      this);
  network_rx_drops_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.system.network.rx_drops.total");
  network_rx_drops_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<SystemMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(observer,
                                                             [](auto p) { return p.rx_drops; });
      },
      this);
  network_tx_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.system.network.tx_bytes.total");
  network_tx_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<SystemMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(observer,
                                                             [](auto p) { return p.tx_bytes; });
      },
      this);
  network_tx_drops_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.system.network.tx_drops.total");
  network_tx_drops_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<SystemMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(observer,
                                                             [](auto p) { return p.tx_drops; });
      },
      this);
}

void SystemMetricsReader::Scrape() {
  // Add CPU metrics for system.
  std::vector<gml::system::ProcParser::CPUStats> stats;
  GML_CHECK_OK(proc_parser_.ParseProcStatAllCPUs(
      &stats, gml::system::Config::GetInstance().KernelTickTimeNS()));
  cpu_stats_ = stats;
  cpu_num_gauge_->Record(static_cast<int64_t>(stats.size()), {{"state", "system"}}, {});

  // Add memory metrics for system.
  gml::system::ProcParser::SystemStats system_stats;
  GML_CHECK_OK(proc_parser_.ParseProcMemInfo(&system_stats));
  mem_stats_total_bytes_->Record(system_stats.mem_total_bytes, {{"state", "system"}}, {});
  mem_stats_free_bytes_->Record(system_stats.mem_free_bytes, {{"state", "system"}}, {});
  mem_stats_buffered_bytes_->Record(system_stats.mem_buffer_bytes, {{"state", "system"}}, {});
  mem_stats_cached_bytes_->Record(system_stats.mem_cached_bytes +
                                      system_stats.mem_sreclaimable_bytes -
                                      system_stats.mem_shmem_bytes,
                                  {{"state", "system"}}, {});

  std::vector<gml::system::ProcParser::NetworkStats> network_stats;
  // This expects --pid=host to be true on the container.
  auto s = proc_parser_.ParseProcHostNetDev(&network_stats);
  if (!s.ok()) {
    LOG(INFO) << "Failed to read proc network stats. Skipping...";
    return;
  }
  network_stats_ = network_stats;

  std::vector<gml::system::CPUFrequencyInfo> freq_stats;
  s = cpu_info_reader_->ReadCPUFrequencies(&freq_stats);
  if (!s.ok()) {
    LOG(INFO) << "Failed to read frequency stats. Skipping... " << s.msg();
    return;
  }

  for (auto f : freq_stats) {
    cpu_frequency_gauge_->Record(f.cpu_freq_hz, {{"cpu", f.cpu_num}}, {});
  }
}

template <typename T>
void SystemMetricsReader::GetObservableResultFromNetworkStats(
    opentelemetry::metrics::ObserverResult observer,
    const std::function<int64_t(const gml::system::ProcParser::NetworkStats& network_stats)>&
        get_stat) {
  for (const auto& [i, n] : Enumerate(network_stats_)) {
    auto val = get_stat(n);
    GetObservableResult<T>(observer)->Observe(val, {{"interface", n.interface}});
  }
}

}  // namespace gml::gem::controller
