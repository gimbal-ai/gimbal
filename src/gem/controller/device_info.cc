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

#include "src/gem/controller/device_info.h"

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/plugins/registry.h"

namespace gml::gem::controller {

using gml::internal::api::core::v1::EDGE_CP_TOPIC_INFO;

controller::DeviceInfoHandler::DeviceInfoHandler(event::Dispatcher* d, GEMInfo* agent_info,
                                                 GRPCBridge* bridge)
    : MessageHandler(d, agent_info, bridge) {}

Status DeviceInfoHandler::SendDeviceInfo() {
  auto caps = capabilities::core::DeviceCapabilities();

  LOG(INFO) << "Collecting device info";
  auto& plugin_registry = plugins::Registry::GetInstance();
  for (auto& name : plugin_registry.RegisteredCapabilityListers()) {
    VLOG(1) << "Listing plugin name: " << name;

    GML_ASSIGN_OR_RETURN(auto builder, plugin_registry.BuildCapabilityLister(name));
    auto s = builder->Populate(&caps);

    if (!s.ok()) {
      LOG(WARNING) << "Failed to get capabilities for lister: " << name << " err " << s.msg();
      continue;
    }
  }
  VLOG(2) << "Device info: " << caps.DebugString();
  VLOG(1) << "Sending device info";

  return bridge()->SendMessageToBridge(EDGE_CP_TOPIC_INFO, caps);
}

Status DeviceInfoHandler::Init() {
  info_timer_ = dispatcher()->CreateTimer([this]() {
    auto s = SendDeviceInfo();
    if (!s.ok()) {
      LOG(ERROR) << "Failed to send info: " << s.msg();
    }
    if (info_timer_) {
      info_timer_->EnableTimer(kInfoInterval);
    }
  });
  info_timer_->EnableTimer(std::chrono::milliseconds(0));
  return Status::OK();
}

Status DeviceInfoHandler::Finish() {
  info_timer_->DisableTimer();
  info_timer_.reset();
  return Status::OK();
}

}  // namespace gml::gem::controller
