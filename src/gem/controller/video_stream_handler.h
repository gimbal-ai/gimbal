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

#include <string>
#include <string_view>

#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/event/dispatcher.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/controller.h"

#include "src/gem/controller/controller.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

class VideoStreamHandler : public MessageHandler {
 public:
  VideoStreamHandler() = delete;
  VideoStreamHandler(gml::event::Dispatcher* d, GEMInfo* info, GRPCBridge* b,
                     exec::core::ControlExecutionContext*);

  ~VideoStreamHandler() override = default;

  Status HandleMessage(const internal::controlplane::egw::v1::BridgeResponse& msg) override;

  Status Init() override;
  Status Finish() override;

 private:
  using H264Chunk = ::gml::internal::api::core::v1::H264Chunk;
  using ImageOverlayChunk = ::gml::internal::api::core::v1::ImageOverlayChunk;

  Status VideoWithOverlaysCallback(const std::vector<ImageOverlayChunk>&,
                                   const std::vector<H264Chunk>&);
  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
  bool running_ = false;
};

}  // namespace gml::gem::controller
