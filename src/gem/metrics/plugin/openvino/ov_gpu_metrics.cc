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

#include "src/gem/metrics/plugin/openvino/ov_gpu_metrics.h"

#include <openvino/runtime/intel_gpu/properties.hpp>

#include "src/common/metrics/metrics_system.h"
#include "src/gem/exec/plugin/openvino/core_singleton.h"
#include "src/gem/metrics/core/scraper_builder.h"
#include "src/gem/metrics/core/shared_metric_names.h"

namespace gml::gem::metrics::openvino {

OpenVinoGPUMetrics::OpenVinoGPUMetrics(gml::metrics::MetricsSystem* metrics_system)
    : core::Scraper(metrics_system) {
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  gpu_mem_usage_gauge_ = gml_meter->CreateInt64Gauge(core::kGPUMemoryGEMUsageGaugeName);
}

void OpenVinoGPUMetrics::Scrape() {
  auto& core = exec::openvino::OpenVinoCoreGetInstance();

  auto ov_devices = core.get_available_devices();

  for (const auto& dev : ov_devices) {
    if (!absl::StartsWith(dev, "GPU")) {
      continue;
    }

    auto device_id = core.get_property(dev, "GPU_DEVICE_ID").as<std::string>();

    auto memory_stats = core.get_property(dev, ov::intel_gpu::memory_statistics.name())
                            .as<decltype(ov::intel_gpu::memory_statistics)::value_type>();

    uint64_t usage_bytes = 0;
    auto it = memory_stats.find("usm_device");
    if (it != memory_stats.end()) {
      usage_bytes = it->second;
    }

    gpu_mem_usage_gauge_->Record(usage_bytes, {{"gpu_id", device_id}}, {});

    VLOG(1) << absl::Substitute("Device $0: OpenVINO GPU memory usage $1", device_id, usage_bytes);
  }
}

};  // namespace gml::gem::metrics::openvino
