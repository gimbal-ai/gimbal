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

#include "src/gem/controller/gem_metrics.h"

#include <unistd.h>

#include "src/common/base/utils.h"
#include "src/common/system/config.h"
#include "src/common/system/proc_parser.h"
#include "src/common/system/proc_pid_path.h"

namespace gml::gem::controller {

template <typename T>
auto GetObservableResult(opentelemetry::metrics::ObserverResult& observer) {
  return std::get<std::shared_ptr<opentelemetry::metrics::ObserverResultT<T>>>(observer);
}

GEMMetricsReader::GEMMetricsReader(::gml::metrics::MetricsSystem* metrics_system)
    : metrics::Scrapeable(metrics_system) {
  CHECK(metrics_system != nullptr);
  pid_ = getpid();

  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  // Setup process stats counters.
  cpu_counter_ = gml_meter->CreateDoubleObservableCounter("gml.gem.cpu.seconds.total");
  cpu_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        absl::base_internal::SpinLockHolder lock(&reader->pid_stats_lock_);
        for (auto& p : reader->pid_process_stats_) {
          auto it = reader->pid_to_tgid_.find(p.first);
          if (it == reader->pid_to_tgid_.end()) {
            LOG(INFO) << "Could not find TGID for PID";
            return;
          }

          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(p.second.utime_ns) / 1E9,
              {{"pid", p.first}, {"state", "user"}, {"tgid", it->second}});
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(p.second.ktime_ns) / 1E9,
              {{"pid", p.first}, {"state", "system"}, {"tgid", it->second}});
        }
      },
      this);
  disk_rchar_counter_ = gml_meter->CreateInt64ObservableCounter("gml.gem.disk.rchar.total");
  disk_rchar_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromProcessStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.rchar_bytes; });
      },
      this);
  disk_wchar_counter_ = gml_meter->CreateInt64ObservableCounter("gml.gem.disk.wchar.total");
  disk_wchar_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromProcessStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.wchar_bytes; });
      },
      this);
  disk_read_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.disk.read_bytes.total");
  disk_read_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromProcessStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.read_bytes; });
      },
      this);
  disk_write_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.disk.write_bytes.total");
  disk_write_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromProcessStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.write_bytes; });
      },
      this);

  mem_usage_gauge_ = gml_meter->CreateInt64Gauge("gml.gem.memory.usage.bytes");
  mem_virtual_gauge_ = gml_meter->CreateInt64Gauge("gml.gem.memory.virtual.bytes");
  thread_gauge_ = gml_meter->CreateInt64Gauge("gml.gem.threads");

  // Setup process status counters.
  context_switches_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.context_switches.total");
  context_switches_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        absl::base_internal::SpinLockHolder lock(&reader->pid_stats_lock_);
        for (auto& p : reader->pid_process_status_) {
          auto it = reader->pid_to_tgid_.find(p.first);
          if (it == reader->pid_to_tgid_.end()) {
            LOG(INFO) << "Could not find TGID for PID";
            return;
          }

          GetObservableResult<int64_t>(observer)->Observe(p.second.voluntary_ctxt_switches,
                                                          {{"context_switch_type", "voluntary"},
                                                           {"pid", p.first},
                                                           {"state", "system"},
                                                           {"tgid", it->second}});
          GetObservableResult<int64_t>(observer)->Observe(p.second.nonvoluntary_ctxt_switches,
                                                          {{"context_switch_type", "involuntary"},
                                                           {"pid", p.first},
                                                           {"state", "system"},
                                                           {"tgid", it->second}});
        }
      },
      this);

  // Setup network stats counters.
  network_rx_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.network.rx_bytes.total");
  network_rx_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.rx_bytes; });
      },
      this);
  network_rx_drops_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.network.rx_drops.total");
  network_rx_drops_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.rx_drops; });
      },
      this);
  network_tx_bytes_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.network.tx_bytes.total");
  network_tx_bytes_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.tx_bytes; });
      },
      this);
  network_tx_drops_counter_ =
      gml_meter->CreateInt64ObservableCounter("gml.gem.network.tx_drops.total");
  network_tx_drops_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto reader = static_cast<GEMMetricsReader*>(parent);
        reader->GetObservableResultFromNetworkStats<int64_t>(std::move(observer),
                                                             [](auto p) { return p.tx_drops; });
      },
      this);
}

void GEMMetricsReader::Scrape() {
  absl::base_internal::SpinLockHolder lock(&pid_stats_lock_);
  // Generate metrics for all child processes in parent.
  auto pids = proc_parser_.ListChildPIDsForPGID(pid_);

  pid_process_stats_.clear();
  pid_network_stats_.clear();
  pid_process_status_.clear();
  pid_to_tgid_.clear();

  for (auto p : pids) {
    auto pid = std::to_string(p);

    // Parse process status.
    gml::system::ProcParser::ProcessStatus process_status;
    auto s = proc_parser_.ParseProcPIDStatus(p, &process_status);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc status. Skipping...";
      continue;
    }
    pid_process_status_[p] = process_status;

    auto tgid = process_status.tgid;
    pid_to_tgid_[p] = tgid;

    // Parse process stats.
    gml::system::ProcParser::ProcessStats process_stats;
    s = proc_parser_.ParseProcPIDStat(p, gml::system::Config::GetInstance().PageSizeBytes(),
                                      gml::system::Config::GetInstance().KernelTickTimeNS(),
                                      &process_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc stats. Skipping...";
      continue;
    }
    s = proc_parser_.ParseProcPIDStatIO(p, &process_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc IO stats. Skipping...";
      continue;
    }
    pid_process_stats_[p] = process_stats;

    mem_usage_gauge_->Record(process_stats.rss_bytes,
                             {{"pid", pid}, {"state", "system"}, {"tgid", tgid}}, {});
    mem_virtual_gauge_->Record(static_cast<int64_t>(process_stats.vsize_bytes),
                               {{"pid", pid}, {"state", "system"}, {"tgid", tgid}}, {});
    thread_gauge_->Record(process_stats.num_threads,
                          {{"pid", pid}, {"state", "system"}, {"tgid", tgid}}, {});

    // Parse network stats.
    std::vector<gml::system::ProcParser::NetworkStats> network_stats;
    s = proc_parser_.ParseProcPIDNetDev(p, &network_stats);
    if (!s.ok()) {
      LOG(INFO) << "Failed to read proc network stats. Skipping...";
      continue;
    }
    pid_network_stats_[p] = network_stats;
  }
}

template <typename T>
void GEMMetricsReader::GetObservableResultFromProcessStats(
    opentelemetry::metrics::ObserverResult observer,
    const std::function<int64_t(const gml::system::ProcParser::ProcessStats& process_stats)>&
        get_stat) {
  absl::base_internal::SpinLockHolder lock(&pid_stats_lock_);
  for (auto& p : pid_process_stats_) {
    auto it = pid_to_tgid_.find(p.first);
    if (it == pid_to_tgid_.end()) {
      LOG(INFO) << "Could not find TGID for PID";
      return;
    }

    auto val = get_stat(p.second);
    GetObservableResult<T>(observer)->Observe(val, {{"pid", p.first}, {"tgid", it->second}});
  }
}

template <typename T>
void GEMMetricsReader::GetObservableResultFromNetworkStats(
    opentelemetry::metrics::ObserverResult observer,
    const std::function<int64_t(const gml::system::ProcParser::NetworkStats& network_stats)>&
        get_stat) {
  absl::base_internal::SpinLockHolder lock(&pid_stats_lock_);
  for (auto& p : pid_network_stats_) {
    auto it = pid_to_tgid_.find(p.first);
    if (it == pid_to_tgid_.end()) {
      LOG(INFO) << "Could not find TGID for PID";
      return;
    }

    for (auto n : p.second) {
      auto val = get_stat(n);
      GetObservableResult<T>(observer)->Observe(
          val, {{"interface", n.interface}, {"pid", p.first}, {"tgid", it->second}});
    }
  }
}

}  // namespace gml::gem::controller
