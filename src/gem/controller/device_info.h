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

#pragma once

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/event/timer.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/gem_info.h"
#include "src/gem/controller/message_handler.h"

namespace gml::gem::controller {

class DeviceInfoHandler : public MessageHandler {
 public:
  static constexpr std::chrono::minutes kInfoInterval = std::chrono::minutes{5};

  DeviceInfoHandler() = delete;
  DeviceInfoHandler(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*);

  ~DeviceInfoHandler() override = default;

  Status HandleMessage(const gml::internal::controlplane::egw::v1::BridgeResponse&) override {
    return Status::OK();
  }

  Status Init() override;
  Status Finish() override;

 private:
  Status SendDeviceInfo();

  event::TimerUPtr info_timer_ = nullptr;
};

}  // namespace gml::gem::controller
