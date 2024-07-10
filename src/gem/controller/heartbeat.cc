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

#include "src/gem/controller/heartbeat.h"

#include <unistd.h>

#include <chrono>

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/event/event.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/uuid/uuid.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/controller/message_handler.h"

namespace gml::gem::controller {

using gml::internal::api::core::v1::EDGE_CP_TOPIC_STATUS;
using gml::internal::api::core::v1::EdgeHeartbeat;
using gml::internal::api::core::v1::EdgeHeartbeatAck;

using gml::internal::controlplane::egw::v1::BridgeResponse;

controller::HeartbeatHandler::HeartbeatHandler(event::Dispatcher* d, GEMInfo* agent_info,
                                               GRPCBridge* bridge)
    : MessageHandler(d, agent_info, bridge),
      time_source_(d->GetTimeSource()),
      heartbeat_send_timer_(dispatcher()->CreateTimer([this] { SendHeartbeat(); })) {
  auto& metrics_system = gml::metrics::MetricsSystem::GetInstance();
  auto meter = metrics_system.GetMeterProvider()->GetMeter("");
  heartbeat_latency_hist_ = meter->CreateUInt64Histogram("heartbeat_latency");
  heartbeat_count_ = meter->CreateUInt64Counter("heartbeat_count");
}

Status HeartbeatHandler::HandleMessage(const BridgeResponse& msg) {
  EdgeHeartbeatAck ack;
  if (!msg.msg().UnpackTo(&ack)) {
    LOG(ERROR) << "Failed to unpack heartbeat ack message. Received message of type: "
               << msg.msg().type_url() << " . Ignoring...";
    return Status::OK();
  }

  auto time_delta = time_source_.MonotonicTime() - heartbeat_info_.last_heartbeat_send_time_;
  heartbeat_latency_hist_->Record(
      std::chrono::duration_cast<std::chrono::milliseconds>(time_delta).count(), {});
  heartbeat_count_->Add(1, {});

  heartbeat_latency_moving_average_ =
      kHbLatencyDecay * heartbeat_latency_moving_average_ + (1 - kHbLatencyDecay) * time_delta;
  LOG_EVERY_N(INFO, 5) << absl::StrFormat(
      "Heartbeat ACK latency moving average: %d ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(heartbeat_latency_moving_average_)
          .count());
  heartbeat_info_.last_ackd_seq_num = ack.seq_id();
  return Status::OK();
}

void HeartbeatHandler::EnableHeartbeats() {
  heartbeat_send_timer_->EnableTimer(std::chrono::milliseconds(0));
}

Status HeartbeatHandler::SendHeartbeatImpl() {
  // HB Code Start.
  if (heartbeat_info_.last_ackd_seq_num < heartbeat_info_.last_sent_seq_num) {
    // Send over the previous request again.
    return bridge()->SendMessageToBridge(EDGE_CP_TOPIC_STATUS, *heartbeat_info_.active_hb_);
  }

  heartbeat_info_.active_hb_ = std::make_unique<EdgeHeartbeat>();
  heartbeat_info_.last_sent_seq_num++;
  heartbeat_info_.active_hb_->set_seq_id(heartbeat_info_.last_sent_seq_num);
  heartbeat_info_.last_heartbeat_send_time_ = time_source_.MonotonicTime();

  return bridge()->SendMessageToBridge(EDGE_CP_TOPIC_STATUS, *heartbeat_info_.active_hb_);
}

void HeartbeatHandler::SendHeartbeat() {
  Status s = SendHeartbeatImpl();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to send heartbeat, will retry on next tick. hb_info: "
               << heartbeat_info_.active_hb_->DebugString() << " error_message: " << s.msg();
  }
  heartbeat_send_timer_->EnableTimer(kHeartbeatInterval);
}

}  // namespace gml::gem::controller
