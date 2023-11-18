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

#include <grpcpp/grpcpp.h>
#include <string>
#include <string_view>

#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/event/timer.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

using gml::internal::controlplane::egw::v1::BridgeRequest;
using gml::internal::controlplane::egw::v1::BridgeResponse;

class MetricsHandler : public MessageHandler {
 public:
  static constexpr std::chrono::seconds kCollectPeriod = std::chrono::seconds{30};
  MetricsHandler() = delete;
  MetricsHandler(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*,
                 exec::core::ControlExecutionContext*);

  ~MetricsHandler() override = default;

  Status HandleMessage(const BridgeResponse&) override { return Status::OK(); }

  Status Init() override;
  Status Finish() override;

 private:
  Status CollectAndPushMetrics();

  event::TimerUPtr collect_timer_ = nullptr;
  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
};

}  // namespace gml::gem::controller
