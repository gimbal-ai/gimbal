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
#include "src/common/system/proc_parser.h"

namespace gml::gem::controller {

SystemMetricsReader::SystemMetricsReader(::gml::metrics::MetricsSystem* metrics_system)
    : metrics::Scrapeable(metrics_system) {
  CHECK(metrics_system != nullptr);
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");
  cpu_stats_counter_ = std::move(gml_meter->CreateInt64UpDownCounter("system.cpu.time"));
}

void SystemMetricsReader::Scrape() {
  std::vector<gml::system::ProcParser::CPUStats> stats;
  GML_CHECK_OK(proc_parser_.ParseProcStatAllCPUs(&stats));

  using kv_t = absl::flat_hash_map<std::string, std::string>;
  kv_t kv;
  for (const auto& [c, stat] : Enumerate(stats)) {
    kv["cpu"] = std::to_string(c);
    kv["state"] = "system";
    cpu_stats_counter_->Add(stat.cpu_ktime_ns,
                            opentelemetry::common::KeyValueIterableView<kv_t>(kv));
    kv["state"] = "user";
    cpu_stats_counter_->Add(stat.cpu_utime_ns,
                            opentelemetry::common::KeyValueIterableView<kv_t>(kv));
  }
}

}  // namespace gml::gem::controller
