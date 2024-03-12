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

#include "src/gem/metrics/plugin/jetson/gpu_metrics.h"

#include "src/common/metrics/metrics_system.h"
#include "src/common/system/nvmap.h"
#include "src/common/system/proc_parser.h"
#include "src/gem/metrics/core/scraper_builder.h"
#include "src/gem/metrics/core/shared_metric_names.h"

namespace gml::gem::metrics::jetson {

JetsonGPUMetrics::JetsonGPUMetrics(gml::metrics::MetricsSystem* metrics_system)
    : core::Scraper(metrics_system) {
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  system_memory_size_gauge_ = gml_meter->CreateInt64Gauge(core::kGPUMemorySystemSizeGaugeName);
  system_memory_usage_gauge_ = gml_meter->CreateInt64Gauge(core::kGPUMemorySystemUsageGaugeName);
  gem_memory_usage_gauge_ = gml_meter->CreateInt64Gauge(core::kGPUMemoryGEMUsageGaugeName);
}

void JetsonGPUMetrics::Scrape() {
  auto s = ScrapeWithError();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to scrape Jetson GPU metrics: " << s.msg();
  }
}

Status JetsonGPUMetrics::ScrapeWithError() {
  // The jetson always has a single integrated GPU, so we give it a fake "gpu_id" to be consistent
  // with other metric plugins.
  static constexpr std::string_view kID = "0x0000";

  std::vector<system::IOVMMClient> iovmm_clients;
  auto path = system::NVMapIOVMMPath() / "clients";
  GML_RETURN_IF_ERROR(system::ParseNVMapIOVMMClients(path, &iovmm_clients));

  uint64_t gem_usage_bytes = 0;
  for (const auto& client : iovmm_clients) {
    if (client.client_type == system::IOVMMClient::IOVMM_CLIENT_TYPE_TOTAL) {
      system_memory_usage_gauge_->Record(client.size_bytes, {{"gpu_id", kID}}, {});
      VLOG(1) << absl::Substitute("System GPU memory usage: $0 MB",
                                  static_cast<double>(client.size_bytes) / 1024. / 1024.);
      continue;
    }
    // TODO(james): use the PID instead of the cmdline. This requires getting the host PID
    // for GEM when it is running inside a container.
    if (client.cmdline != "gem") {
      continue;
    }
    gem_usage_bytes = client.size_bytes;
  }
  gem_memory_usage_gauge_->Record(gem_usage_bytes, {{"gpu_id", kID}}, {});
  VLOG(1) << absl::Substitute("GEM GPU memory usage: $0 MB",
                              static_cast<double>(gem_usage_bytes) / 1024. / 1024.);

  system::ProcParser::SystemStats stats = {};
  GML_RETURN_IF_ERROR(proc_parser_.ParseProcMemInfo(&stats));
  system_memory_size_gauge_->Record(stats.mem_total_bytes, {{"gpu_id", kID}}, {});

  return Status::OK();
}

}  // namespace gml::gem::metrics::jetson
