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

#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/gem_info.h"
#include "src/gem/controller/message_handler.h"

namespace gml::gem::controller {

class HeartbeatHandler : public MessageHandler {
 public:
  HeartbeatHandler() = delete;
  HeartbeatHandler(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*);

  ~HeartbeatHandler() override = default;

  Status Init() override { return Status::OK(); }

  Status HandleMessage(const gml::internal::controlplane::egw::v1::BridgeResponse& msg) override;

  Status Finish() override { return Status::OK(); }

  void EnableHeartbeats();

 private:
  struct HeartbeatInfo {
    HeartbeatInfo() : active_hb_(std::make_unique<gml::internal::api::core::v1::EdgeHeartbeat>()) {}

    int64_t last_sent_seq_num = -1;
    int64_t last_ackd_seq_num = -1;
    std::chrono::steady_clock::time_point last_heartbeat_send_time_;
    std::chrono::duration<double> heartbeat_latency_moving_average_{0};
    std::unique_ptr<gml::internal::api::core::v1::EdgeHeartbeat> active_hb_;
  };

  HeartbeatInfo heartbeat_info_;
  const gml::event::TimeSource& time_source_;
  Status SendHeartbeatImpl();
  void SendHeartbeat();

  gml::event::TimerUPtr heartbeat_send_timer_;
  std::chrono::duration<double> heartbeat_latency_moving_average_{0};

  std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> heartbeat_latency_hist_;
  std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> heartbeat_count_;

  static constexpr double kHbLatencyDecay = 0.25;

  static constexpr std::chrono::seconds kHeartbeatInterval{2};
  static constexpr int kHeartbeatRetryCount = 5;

  // TODO(zasgar): Implement heartbeat watch dog.
  // gml::event::TimerUPtr heartbeat_watchdog_timer_;
  // static constexpr std::chrono::milliseconds kHeartbeatWaitMillis{5000};
};

}  // namespace gml::gem::controller
