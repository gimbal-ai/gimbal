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

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <sole.hpp>
#include <tbb/concurrent_queue.h>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
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

  void HandleReadWriteFailure() ABSL_LOCKS_EXCLUDED(rdwr_lock_);
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

  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> rx_msg_total_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> tx_msg_total_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> rx_err_total_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> tx_err_total_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> rx_msg_bytes_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> tx_msg_bytes_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> stream_reset_total_;
  ;
};

}  // namespace gml::gem::controller
