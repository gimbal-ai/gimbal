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

#include "src/gem/metrics/plugin/openvino/ov_gpu_metrics.h"

#include <openvino/runtime/intel_gpu/properties.hpp>

#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/exec/plugin/openvino/core_singleton.h"
#include "src/gem/metrics/core/scraper_builder.h"
#include "src/gem/metrics/core/shared_metric_names.h"

namespace gml::gem::metrics::openvino {

OpenVinoGPUMetrics::OpenVinoGPUMetrics(gml::metrics::MetricsSystem* metrics_system,
                                       gml::event::Dispatcher*)
    : core::Scraper(metrics_system) {
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  gpu_mem_usage_gauge_ =
      gml_meter->CreateInt64Gauge(core::kGPUMemoryGEMUsageGaugeName,
                                  "The total number of bytes of GPU memory used by the GEM.");
}

void OpenVinoGPUMetrics::Scrape() {
  auto& core = exec::openvino::OpenVinoCoreGetInstance();

  auto ov_devices = core.get_available_devices();

  for (const auto& dev : ov_devices) {
    if (!absl::StartsWith(dev, "GPU")) {
      continue;
    }

    auto device_uuid = core.get_property(dev, ov::device::uuid);
    std::stringstream device_id_ss;
    device_id_ss << device_uuid;
    auto device_id = device_id_ss.str();

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
