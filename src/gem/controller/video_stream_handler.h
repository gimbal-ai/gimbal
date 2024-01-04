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
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

class VideoStreamHandler : public MessageHandler {
 public:
  static constexpr std::chrono::seconds kKeepAliveInterval = std::chrono::seconds{5};

  VideoStreamHandler() = delete;
  VideoStreamHandler(gml::event::Dispatcher* d, GEMInfo* info, GRPCBridge* b,
                     exec::core::ControlExecutionContext*);

  ~VideoStreamHandler() override = default;

  Status Start();

  Status HandleMessage(const internal::controlplane::egw::v1::BridgeResponse& msg) override;

  Status Init() override;
  Status Finish() override;

 private:
  Status VideoWithOverlaysCallback(const std::vector<std::unique_ptr<google::protobuf::Message>>&);
  exec::core::ControlExecutionContext* ctrl_exec_ctx_;

  event::TimerUPtr keep_alive_timer_ = nullptr;
  bool running_ = false;
};

}  // namespace gml::gem::controller
