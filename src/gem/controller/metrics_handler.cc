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

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/metrics_handler.h"
#include "src/gem/exec/core/control_context.h"

using gml::internal::api::core::v1::EDGE_CP_TOPIC_METRICS;
using ::gml::internal::api::core::v1::EdgeOTelMetrics;

namespace gml::gem::controller {

Status MetricsHandler::CollectAndPushMetrics() {
  GML_UNUSED(ctrl_exec_ctx_);
  auto& metrics_system = gml::metrics::MetricsSystem::GetInstance();
  auto resource_metrics = metrics_system.CollectAllAsProto();
  EdgeOTelMetrics metrics;
  (*metrics.mutable_resource_metrics()) = resource_metrics;
  return bridge()->SendMessageToBridge(EDGE_CP_TOPIC_METRICS, metrics);
}

Status MetricsHandler::Init() {
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
  collect_timer_->EnableTimer(kCollectPeriod);
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
