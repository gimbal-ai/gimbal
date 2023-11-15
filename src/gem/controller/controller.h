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

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/common/base/base.h"
#include "src/common/event/event.h"
#include "src/common/system/system.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/gem/controller/message_handler.h"

namespace gml::gem::controller {

class Controller : public gml::NotCopyable {
 public:
  Controller() = delete;
  Controller(std::string_view deploy_key, std::string_view cp_addr)
      : deploy_key_(std::string(deploy_key)),
        cp_addr_(std::string(cp_addr)),
        time_system_(std::make_unique<gml::event::RealTimeSystem>()),
        api_(std::make_unique<gml::event::APIImpl>(time_system_.get())),
        dispatcher_(api_->AllocateDispatcher("controller")) {}

  virtual ~Controller() = default;

  Status Init();

  Status Run();

  Status Stop(std::chrono::milliseconds timeout);

 protected:
  gml::event::Dispatcher* dispatcher() { return dispatcher_.get(); }
  Status RegisterMessageHandler(gml::internal::api::core::v1::CPEdgeTopic,
                                std::shared_ptr<MessageHandler> handler);
  Status HandleMessage(std::unique_ptr<gml::internal::controlplane::egw::v1::BridgeResponse> msg);

 private:
  // Marks if the controller is still running. Force stopping will cause un-graceful termination.
  std::atomic<bool> running_ = false;
  std::string deploy_key_;
  GEMInfo info_;
  std::string cp_addr_;
  // The time system to use (real or simulated).
  std::unique_ptr<gml::event::TimeSystem> time_system_;
  gml::event::APIUPtr api_;

  gml::event::DispatcherUPtr dispatcher_ = nullptr;
  std::unique_ptr<GRPCBridge> bridge_;

  std::shared_ptr<grpc::Channel> cp_chan_;
  std::unique_ptr<gml::internal::controlplane::fleetmgr::v1::FleetMgrEdgeService::Stub> fmstub_;
  std::unique_ptr<gml::internal::controlplane::egw::v1::EGWService::Stub> egwstub_;
  absl::flat_hash_map<gml::internal::api::core::v1::CPEdgeTopic, std::shared_ptr<MessageHandler>>
      message_handlers_;
};

}  // namespace gml::gem::controller
