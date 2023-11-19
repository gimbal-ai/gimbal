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

#include <grpcpp/grpcpp.h>
#include <condition_variable>
#include <string>
#include <string_view>

#include <tbb/concurrent_queue.h>
#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"

namespace gml::gem::controller {

class GRPCBridge {
 public:
  using BridgeRequest = ::gml::internal::controlplane::egw::v1::BridgeRequest;
  using BridgeResponse = ::gml::internal::controlplane::egw::v1::BridgeResponse;

  GRPCBridge() = delete;
  GRPCBridge(std::shared_ptr<grpc::Channel> cp_chan, std::string_view deploy_key,
             const sole::uuid& device_id);
  virtual ~GRPCBridge() = default;

  Status Init();

  // Run is a non-blocking function that starts the GRPC bridge.
  Status Run();

  // Shutdown needs to be called for a graceful termination of the GRPC bridge.
  Status Shutdown();

  // Enqueue a message to be sent on the GRPC bridge. This can be called from any thread, but
  // messages are generally written on sequentially to avoid issues since gRPC can only have a
  // single outstanding read/write.
  Status SendMessageToBridge(gml::internal::api::core::v1::EdgeCPTopic,
                             const google::protobuf::Message& msg);

  // Register a message handler to be called on message read. This is called on the reader thread
  // and should not block.
  void RegisterOnMessageReadHandler(
      std::function<Status(std::unique_ptr<gml::internal::controlplane::egw::v1::BridgeResponse>)>
          handler) {
    read_handler_ = std::move(handler);
  }

 private:
  void Reader();
  void Writer();

  Status WriteWithRetries(const BridgeRequest&);
  Status WriteRequestToBridge(const BridgeRequest&);

  Status HandleReadWriteFailure() ABSL_LOCKS_EXCLUDED(rdwr_lock_);
  Status Connect() ABSL_EXCLUSIVE_LOCKS_REQUIRED(rdwr_lock_);

  std::string controlplane_addr_;
  std::shared_ptr<grpc::Channel> cp_chan_;
  std::string deploy_key_;
  sole::uuid device_id_;
  std::atomic<bool> running_;

  tbb::concurrent_bounded_queue<std::unique_ptr<BridgeRequest>> write_q_;

  std::unique_ptr<std::thread> read_thread_;
  std::unique_ptr<std::thread> write_thread_;

  std::unique_ptr<gml::internal::controlplane::egw::v1::EGWService::Stub> egwstub_;
  std::unique_ptr<::grpc::ClientReaderWriter<BridgeRequest, BridgeResponse>> rdwr_
      ABSL_GUARDED_BY(rdwr_lock_) ABSL_PT_GUARDED_BY(rdwr_lock_);
  absl::Mutex rdwr_lock_;

  std::unique_ptr<grpc::ClientContext> ctx_;
  std::function<Status(std::unique_ptr<BridgeResponse>)> read_handler_;
};

}  // namespace gml::gem::controller
