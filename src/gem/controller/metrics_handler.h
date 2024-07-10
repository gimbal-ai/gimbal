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

#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/event/timer.h"
#include "src/common/metrics/metrics_system.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

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

  std::vector<std::unique_ptr<metrics::Scrapeable>> metrics_scrapers_;
};

}  // namespace gml::gem::controller
