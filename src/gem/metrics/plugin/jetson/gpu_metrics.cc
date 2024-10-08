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

#include "src/gem/metrics/plugin/jetson/gpu_metrics.h"

#include <chrono>

#include "src/common/base/error.h"
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/system/config.h"
#include "src/common/system/nvmap.h"
#include "src/common/system/proc_parser.h"
#include "src/gem/metrics/core/scraper_builder.h"
#include "src/gem/metrics/core/shared_metric_names.h"

namespace gml::gem::metrics::jetson {

constexpr auto kCollectUtilizationPeriod = std::chrono::milliseconds{10};
// The jetson always has a single integrated GPU, so we give it a fake "gpu_id" to be consistent
// with other metric plugins.
static constexpr std::string_view kID = "0x0000";

namespace {

template <typename T>
auto GetObservableResult(opentelemetry::metrics::ObserverResult& observer) {
  return std::get<std::shared_ptr<opentelemetry::metrics::ObserverResultT<T>>>(observer);
}

}  // namespace

JetsonGPUMetrics::JetsonGPUMetrics(gml::metrics::MetricsSystem* metrics_system,
                                   gml::event::Dispatcher* dispatcher)
    : core::Scraper(metrics_system), dispatcher_(dispatcher) {
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  system_memory_size_gauge_ = gml_meter->CreateInt64Gauge(
      core::kGPUMemorySystemSizeGaugeName, "The size of GPU memory on the device in bytes.");
  system_memory_usage_gauge_ =
      gml_meter->CreateInt64Gauge(core::kGPUMemorySystemUsageGaugeName,
                                  "The total number of bytes of GPU memory inuse on the device.");
  gem_memory_usage_gauge_ =
      gml_meter->CreateInt64Gauge(core::kGPUMemoryGEMUsageGaugeName,
                                  "The total number of bytes of GPU memory inuse by the GEM.");
  system_utilization_counter_ = gml_meter->CreateDoubleObservableCounter(
      core::kGPUUtilizationSystemCounterName,
      "The total GPU time consumed by the system in seconds.");
  system_utilization_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto gpu_metrics = static_cast<JetsonGPUMetrics*>(parent);
        absl::base_internal::SpinLockHolder lock(&gpu_metrics->utilization_lock_);

        GetObservableResult<double>(observer)->Observe(
            static_cast<double>(gpu_metrics->system_utilization_ns_) / 1E9, {
                                                                                {"gpu_id", kID},
                                                                            });
      },
      this);

  auto s = InitUtilizationCollection();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to initialize jetson GPU utilization collection: " << s.msg();
  }
}

void JetsonGPUMetrics::Scrape() {
  auto s = ScrapeWithError();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to scrape Jetson GPU metrics: " << s.msg();
  }
}

Status JetsonGPUMetrics::ScrapeWithError() {
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

Status JetsonGPUMetrics::InitUtilizationCollection() {
  static constexpr std::string_view kLoadRelPath = "devices/platform/gpu.0/load";
  static constexpr int kMaxLoadFileChars = 4;

  auto load_path =
      std::filesystem::path(gml::system::Config::GetInstance().sys_path()) / kLoadRelPath;

  load_file_.open(load_path);
  if (!load_file_.is_open()) {
    return error::Internal("failed to open GPU load file $0", load_path.string());
  }

  util_collect_timer_ = dispatcher_->CreateTimer([this]() {
    CollectUtilization();
    if (util_collect_timer_) {
      util_collect_timer_->EnableTimer(kCollectUtilizationPeriod);
    }
  });
  prev_collect_ = std::chrono::high_resolution_clock::now();
  util_collect_timer_->EnableTimer(std::chrono::milliseconds(0));

  line_buf_.reserve(kMaxLoadFileChars + 1);

  return Status::OK();
}

void JetsonGPUMetrics::CollectUtilization() {
  line_buf_.resize(0);
  std::getline(load_file_, line_buf_);
  auto now = std::chrono::high_resolution_clock::now();
  load_file_.seekg(0, std::ios::beg);

  uint32_t load = 0;
  if (!absl::SimpleAtoi(line_buf_, &load)) {
    return;
  }

  uint64_t dur_nanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - prev_collect_).count();
  // 1 unit of load represents 0.1% utilization.
  uint64_t increment = dur_nanos * load / 1000;
  prev_collect_ = now;

  absl::base_internal::SpinLockHolder lock(&utilization_lock_);
  system_utilization_ns_ += increment;
}

}  // namespace gml::gem::metrics::jetson
