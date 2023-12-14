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
