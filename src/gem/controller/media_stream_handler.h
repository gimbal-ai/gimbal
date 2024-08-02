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

class MediaStreamHandler : public MessageHandler {
 public:
  static constexpr std::chrono::seconds kKeepAliveInterval = std::chrono::seconds{5};

  MediaStreamHandler() = delete;
  MediaStreamHandler(gml::event::Dispatcher* d, GEMInfo* info, GRPCBridge* b,
                     exec::core::ControlExecutionContext*);

  ~MediaStreamHandler() override = default;

  Status Start();

  Status HandleMessage(const internal::controlplane::egw::v1::BridgeResponse& msg) override;

  Status Init() override;
  Status Finish() override;

 private:
  Status MediaStreamCallback(const std::vector<std::unique_ptr<google::protobuf::Message>>&);
  exec::core::ControlExecutionContext* ctrl_exec_ctx_;

  event::TimerUPtr keep_alive_timer_ = nullptr;
  bool running_ = false;
};

}  // namespace gml::gem::controller
