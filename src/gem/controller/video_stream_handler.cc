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

#include "src/gem/controller/video_stream_handler.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <grpcpp/grpcpp.h>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/event/event.h"
#include "src/common/uuid/uuid.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/exec/core/control_context.h"

using gml::internal::api::core::v1::EDGE_CP_TOPIC_VIDEO;

namespace gml::gem::controller {

using ::gml::internal::api::core::v1::VideoStreamKeepAlive;
using ::gml::internal::api::core::v1::VideoStreamStart;
using ::gml::internal::api::core::v1::VideoStreamStop;

VideoStreamHandler::VideoStreamHandler(gml::event::Dispatcher* d, GEMInfo* info, GRPCBridge* b,
                                       exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(d, info, b), ctrl_exec_ctx_(ctrl_exec_ctx) {}

Status VideoStreamHandler::VideoWithOverlaysCallback(
    const std::vector<std::unique_ptr<google::protobuf::Message>>& messages) {
  for (const auto& message : messages) {
    GML_RETURN_IF_ERROR(bridge()->SendMessageToBridge(EDGE_CP_TOPIC_VIDEO, *message));
  }
  return Status::OK();
}

Status VideoStreamHandler::Start() {
  LOG(INFO) << "Starting VideoStreamHandler";
  ctrl_exec_ctx_->RegisterMediaStreamCallback(
      std::bind(&VideoStreamHandler::VideoWithOverlaysCallback, this, std::placeholders::_1));
  running_ = true;

  keep_alive_timer_ = dispatcher()->CreateTimer([this]() {
    VLOG(1) << "Missed VideoStreamKeepAlive message. Stopping video stream.";
    auto s = Finish();
    if (!s.ok()) {
      LOG(ERROR) << "Failed to stop stream: " << s.msg();
    }
  });
  keep_alive_timer_->EnableTimer(kKeepAliveInterval);

  return Status::OK();
}

Status VideoStreamHandler::HandleMessage(
    const gml::internal::controlplane::egw::v1::BridgeResponse& msg) {
  if (msg.msg().Is<VideoStreamStop>()) {
    return Finish();
  }

  if (msg.msg().Is<VideoStreamStart>() || msg.msg().Is<VideoStreamKeepAlive>()) {
    if (!running_) {
      return Start();
    }
    if (keep_alive_timer_) {
      VLOG(1) << "Resetting KeepAlive timer for VideoStreamHandler";
      // Already running, just reset the timer.
      keep_alive_timer_->EnableTimer(kKeepAliveInterval);
      return Status::OK();
    }
  }

  LOG(ERROR) << "Failed to unpack VideoStreamMessage. Recived message of type: "
             << msg.msg().type_url() << " . Ignoring...";
  return Status::OK();
}

Status VideoStreamHandler::Init() { return Status::OK(); }

Status VideoStreamHandler::Finish() {
  LOG(INFO) << "Stopping VideoStreamHandler";
  if (keep_alive_timer_) {
    keep_alive_timer_->DisableTimer();
    keep_alive_timer_.reset();
  }
  running_ = false;
  ctrl_exec_ctx_->ClearMediaStreamCallback();
  return Status::OK();
}

}  // namespace gml::gem::controller
