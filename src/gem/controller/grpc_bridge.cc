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

#include "src/gem/controller/grpc_bridge.h"

#include <memory>

namespace gml::gem::controller {

using gml::internal::controlplane::egw::v1::BridgeRequest;
using gml::internal::controlplane::egw::v1::BridgeResponse;

using gml::internal::controlplane::egw::v1::EGWService;

GRPCBridge::GRPCBridge(std::shared_ptr<grpc::Channel> cp_chan, std::string_view deploy_key,
                       const sole::uuid& device_id)
    : cp_chan_(std::move(cp_chan)), deploy_key_(std::string(deploy_key)), device_id_(device_id) {}

Status GRPCBridge::Init() {
  egwstub_ = std::make_unique<EGWService::Stub>(cp_chan_);

  ctx.AddMetadata("x-deploy-key", deploy_key_);
  ctx.AddMetadata("x-device-id", device_id_.str());
  rdwr_ = egwstub_->Bridge(&ctx);
  if (rdwr_ == nullptr) {
    return error::Internal("Failed to connect to controlplane bridge.");
  }
  return Status::OK();
}
Status GRPCBridge::Run() {
  running_ = true;
  // Create  thread to read message.
  read_thread_ = std::make_unique<std::thread>(&GRPCBridge::Reader, this);
  write_thread_ = std::make_unique<std::thread>(&GRPCBridge::Writer, this);

  return Status::OK();
}

Status GRPCBridge::Shutdown() {
  running_ = false;
  // Stop the writer thread.
  write_q_.emplace(nullptr);

  // Try to close the connection.
  ctx.TryCancel();

  if (rdwr_) {
    rdwr_->Finish();
  }
  if (read_thread_) {
    read_thread_->join();
  }
  if (write_thread_) {
    write_thread_->join();
  }
  return Status::OK();
}

void GRPCBridge::Reader() {
  while (running_) {
    auto resp = std::make_unique<BridgeResponse>();
    if (!rdwr_->Read(resp.get())) {
      LOG(ERROR) << "Failed to read message";
      // TODO(zasgar): We need to notify of error here.
      return;
    }

    if (read_handler_) {
      ECHECK_OK(read_handler_(std::move(resp)));
    }
  }
}

void GRPCBridge::Writer() {
  while (running_) {
    std::unique_ptr<BridgeRequest> req;
    if (!write_q_.try_pop(req)) {
      continue;
    }
    if (req == nullptr) {
      // Got the sentinel value saying the thread should be terminated.
      return;
    }
    if (!rdwr_->Write(*req)) {
      LOG(ERROR) << "Failed to read message";
      // TODO(zasgar): We need to notify of error here.
      return;
    }
  }
}

Status GRPCBridge::SendMessageToBridge(gml::internal::api::core::v1::EdgeCPTopic topic,
                                       const google::protobuf::Message& msg) {
  // Wrap outgoing message.
  auto wrapper = std::make_unique<BridgeRequest>();
  wrapper->set_topic(topic);
  wrapper->mutable_msg()->PackFrom(msg);

  write_q_.emplace(std::move(wrapper));
  return Status::OK();
}

}  // namespace gml::gem::controller
