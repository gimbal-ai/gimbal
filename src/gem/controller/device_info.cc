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

#include "src/gem/controller/device_info.h"
#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/plugins/registry.h"

namespace gml::gem::controller {

using gml::internal::api::core::v1::EDGE_CP_TOPIC_INFO;
using gml::internal::controlplane::egw::v1::BridgeResponse;

controller::DeviceInfoHandler::DeviceInfoHandler(event::Dispatcher* d, GEMInfo* agent_info,
                                                 GRPCBridge* bridge)
    : MessageHandler(d, agent_info, bridge) {}

Status DeviceInfoHandler::SendDeviceInfo() {
  auto caps = capabilities::core::DeviceCapabilities();

  auto& plugin_registry = plugins::Registry::GetInstance();
  for (auto& name : plugin_registry.RegisteredCapabilityListers()) {
    GML_ASSIGN_OR_RETURN(auto builder, plugin_registry.BuildCapabilityLister(name));
    auto s = builder->Populate(&caps);

    if (!s.ok()) {
      return s;
    }
  }

  return bridge()->SendMessageToBridge(EDGE_CP_TOPIC_INFO, caps);
}

Status DeviceInfoHandler::Init() {
  info_timer_ = dispatcher()->CreateTimer([this]() {
    VLOG(1) << "Sending device info";
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
