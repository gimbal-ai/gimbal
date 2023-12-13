
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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <grpcpp/grpcpp.h>
#include <chrono>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/event/event.h"
#include "src/common/uuid/uuid.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/controller/video_stream_handler.h"
#include "src/gem/exec/core/control_context.h"

using gml::internal::api::core::v1::EDGE_CP_TOPIC_VIDEO;

namespace gml::gem::controller {

using ::gml::internal::api::core::v1::H264Chunk;
using ::gml::internal::api::core::v1::ImageOverlayChunk;

VideoStreamHandler::VideoStreamHandler(gml::event::Dispatcher* d, GEMInfo* info, GRPCBridge* b,
                                       exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(d, info, b), ctrl_exec_ctx_(ctrl_exec_ctx) {}

Status VideoStreamHandler::VideoWithOverlaysCallback(
    const std::vector<ImageOverlayChunk>& image_overlay_chunks,
    const std::vector<H264Chunk>& h264_chunks) {
  for (const auto& chunk : image_overlay_chunks) {
    GML_RETURN_IF_ERROR(bridge()->SendMessageToBridge(EDGE_CP_TOPIC_VIDEO, chunk));
  }
  for (const auto& chunk : h264_chunks) {
    GML_RETURN_IF_ERROR(bridge()->SendMessageToBridge(EDGE_CP_TOPIC_VIDEO, chunk));
  }
  return Status::OK();
}

Status VideoStreamHandler::HandleMessage(
    const gml::internal::controlplane::egw::v1::BridgeResponse&) {
  if (running_) {
    return Status::OK();
  }

  LOG(INFO) << "Starting VideoStreamHandler";
  ctrl_exec_ctx_->RegisterVideoWithOverlaysCallback(
      std::bind(&VideoStreamHandler::VideoWithOverlaysCallback, this, std::placeholders::_1,
                std::placeholders::_2));
  running_ = true;

  return Status::OK();
}

Status VideoStreamHandler::Init() { return Status::OK(); }

Status VideoStreamHandler::Finish() {
  ctrl_exec_ctx_->ClearVideoWithOverlaysCallback();
  return Status::OK();
}

}  // namespace gml::gem::controller
