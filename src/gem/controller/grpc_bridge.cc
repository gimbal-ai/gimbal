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

#include <magic_enum.hpp>
#include <memory>
#include "src/common/base/error.h"
#include "src/common/grpcutils/status.h"

namespace gml::gem::controller {

using gml::internal::controlplane::egw::v1::BridgeRequest;
using gml::internal::controlplane::egw::v1::BridgeResponse;

using gml::internal::controlplane::egw::v1::EGWService;

GRPCBridge::GRPCBridge(std::shared_ptr<grpc::Channel> cp_chan, std::string_view deploy_key,
                       const sole::uuid& device_id)
    : cp_chan_(std::move(cp_chan)), deploy_key_(std::string(deploy_key)), device_id_(device_id) {}

Status GRPCBridge::Connect() {
  ctx_ = std::make_unique<grpc::ClientContext>();

  ctx_->AddMetadata("x-deploy-key", deploy_key_);
  ctx_->AddMetadata("x-device-id", device_id_.str());

  rdwr_ = egwstub_->Bridge(ctx_.get());
  if (rdwr_ == nullptr) {
    return error::Internal("Failed to connect to controlplane bridge.");
  }
  return Status::OK();
}

Status GRPCBridge::HandleReadWriteFailure() {
  absl::WriterMutexLock lock(&rdwr_lock_);
  rdwr_->WritesDone();
  auto grpc_status = rdwr_->Finish();
  // Check if the error is a RST_STREAM.
  if (grpc_status.ok() || grpc_status.error_code() != grpc::StatusCode::INTERNAL ||
      !absl::StrContains(grpc_status.error_message(), "RST_STREAM")) {
    if (grpc_status.ok()) {
      // Successful Finish call implies the server cancelled the stream without error.
      return error::Cancelled("GRPCBridge stream cancelled by server");
    }
    return StatusAdapter(grpc_status);
  }
  // If it is a RST_STREAM restart the stream.
  LOG(INFO) << "Restarting GRPCBridge due to RST_STREAM";

  stream_reset_total_->Add(1);

  GML_RETURN_IF_ERROR(Connect());
  return Status::OK();
}

Status GRPCBridge::Init() {
  egwstub_ = std::make_unique<EGWService::Stub>(cp_chan_);
  absl::WriterMutexLock lock(&rdwr_lock_);
  GML_RETURN_IF_ERROR(Connect());

  auto& metrics_system = gml::metrics::MetricsSystem::GetInstance();
  auto meter = metrics_system.GetMeterProvider()->GetMeter("gml");

  rx_msg_total_ = meter->CreateUInt64Counter(
      "gml.gem.bridge.rx_msg_total", "The total number of messages received by the GEM over GRPC");
  tx_msg_total_ = meter->CreateUInt64Counter(
      "gml.gem.bridge.tx_msg_total", "The total number of messages sent by the GEM over GRPC");
  rx_err_total_ = meter->CreateUInt64Counter("gml.gem.bridge.rx_err_total",
                                             "The total number of read errors on GEM bridge");
  tx_err_total_ = meter->CreateUInt64Counter("gml.gem.bridge.tx_err_total",
                                             "The total number of write errors on GEM bridge");
  rx_msg_bytes_ = meter->CreateUInt64Counter(
      "gml.gem.bridge.rx_msg_bytes", "The total number of bytes received by the GEM over GRPC");
  tx_msg_bytes_ = meter->CreateUInt64Counter("gml.gem.bridge.tx_msg_bytes",
                                             "The total number of bytes sent by the GEM over GRPC");

  stream_reset_total_ = meter->CreateUInt64Counter("gml.gem.bridge.stream_reset_total",
                                                   "The number of GRPC stream resets");

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

  {
    absl::WriterMutexLock lock(&rdwr_lock_);
    if (rdwr_) {
      rdwr_->WritesDone();
      rdwr_->Finish();
    }
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
    bool read_succeeded;
    {
      absl::ReaderMutexLock lock(&rdwr_lock_);
      read_succeeded = rdwr_->Read(resp.get());
    }
    if (!read_succeeded) {
      rx_err_total_->Add(1);
      auto s = HandleReadWriteFailure();
      if (!s.ok()) {
        // TODO(zasgar): We need to notify of error here.
        LOG(FATAL) << absl::Substitute("GRPCBridge read failed with error: $0", s.msg());
      }
    }

    std::string topic(magic_enum::enum_name(resp->topic()));

    rx_msg_total_->Add(1, {{"topic", topic}});
    rx_msg_bytes_->Add(resp->ByteSizeLong(), {{"topic", topic}});

    if (read_succeeded && read_handler_) {
      ECHECK_OK(read_handler_(std::move(resp)));
    }
  }
}

Status GRPCBridge::WriteRequestToBridge(const BridgeRequest& req) {
  bool write_suceeded;
  {
    absl::ReaderMutexLock lock(&rdwr_lock_);
    write_suceeded = rdwr_->Write(req);
  }
  if (!write_suceeded) {
    tx_err_total_->Add(1);
    GML_RETURN_IF_ERROR(HandleReadWriteFailure());
  }

  std::string topic(magic_enum::enum_name(req.topic()));
  tx_msg_total_->Add(1, {{"topic", topic}});
  tx_msg_bytes_->Add(req.ByteSizeLong(), {{"topic", topic}});

  return Status::OK();
}

Status GRPCBridge::WriteWithRetries(const BridgeRequest& req) {
  constexpr int kWriteRetries = 3;

  Status s;
  for (int i = 0; i < kWriteRetries; ++i) {
    s = WriteRequestToBridge(req);
    if (s.ok()) {
      return Status::OK();
    }
  }
  return error::Internal("Failed to write request ($0) to bridge after $1 retries, giving up...",
                         s.msg(), kWriteRetries);
}

void GRPCBridge::Writer() {
  while (running_) {
    std::unique_ptr<BridgeRequest> req;
    write_q_.pop(req);
    if (req == nullptr) {
      // Got the sentinel value saying the thread should be terminated.
      return;
    }
    auto s = WriteWithRetries(*req);
    if (!s.ok()) {
      // TODO(zasgar): We need to notify of error here.
      LOG(FATAL) << "GRPCBridge Writer failed: " << s.msg();
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
