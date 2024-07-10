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

#include "src/gem/controller/metrics_handler.h"

#include <unistd.h>

#include <chrono>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/plugins/registry.h"

DEFINE_int64(metrics_chunk_size_bytes,
             gflags::Int64FromEnv("GML_METRICS_CHUNK_SIZE", 1024UL * 512UL),
             "The chunk size for the metrics we send out, in bytes.");

using gml::internal::api::core::v1::EDGE_CP_TOPIC_METRICS;
using ::gml::internal::api::core::v1::EdgeOTelMetrics;

namespace gml::gem::controller {

Status MetricsHandler::CollectAndPushMetrics() {
  GML_UNUSED(ctrl_exec_ctx_);
  auto& metrics_system = gml::metrics::MetricsSystem::GetInstance();

  // Trigger update of stats that require polling.
  for (auto& s : metrics_system.scrapeables()) {
    s->Scrape();
  }

  // Collect all OTel metrics.
  auto resource_metrics = metrics_system.CollectAllAsProto();

  // We have auxiliary stats providers, like MediaPipe, so collect those stats as well.
  for (auto& s : metrics_system.aux_metrics_providers()) {
    GML_RETURN_IF_ERROR(s->CollectMetrics(&resource_metrics));
  }

  auto chunked_metrics =
      metrics_system.ChunkMetrics(&resource_metrics, FLAGS_metrics_chunk_size_bytes);
  for (const auto& metric : chunked_metrics) {
    EdgeOTelMetrics metrics;
    *metrics.mutable_resource_metrics() = metric;
    GML_RETURN_IF_ERROR(bridge()->SendMessageToBridge(EDGE_CP_TOPIC_METRICS, metrics));
  }

  return Status::OK();
}

Status MetricsHandler::Init() {
  auto& plugin_registry = plugins::Registry::GetInstance();
  auto& metrics_system = gml::metrics::MetricsSystem::GetInstance();
  for (const auto& scraper_name : plugin_registry.RegisteredMetricsScrapers()) {
    GML_ASSIGN_OR_RETURN(auto scraper, plugin_registry.BuildMetricsScraper(
                                           scraper_name, &metrics_system, dispatcher()));
    metrics_scrapers_.push_back(std::move(scraper));
  }
  collect_timer_ = dispatcher()->CreateTimer([this]() {
    VLOG(1) << "Collecting GEM metrics";
    auto s = CollectAndPushMetrics();
    if (!s.ok()) {
      LOG(ERROR) << "Failed to collect metrics: " << s.msg();
    }
    if (collect_timer_) {
      collect_timer_->EnableTimer(kCollectPeriod);
    }
  });
  collect_timer_->EnableTimer(std::chrono::milliseconds(0));
  return Status::OK();
}

Status MetricsHandler::Finish() {
  collect_timer_->DisableTimer();
  collect_timer_.reset();
  return Status::OK();
}

MetricsHandler::MetricsHandler(gml::event::Dispatcher* dispatcher, GEMInfo* info,
                               GRPCBridge* bridge,
                               exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(dispatcher, info, bridge), ctrl_exec_ctx_(ctrl_exec_ctx) {}

}  // namespace gml::gem::controller
