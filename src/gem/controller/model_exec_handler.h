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
#include "src/common/event/task.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/message_handler.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/exec/core/runner/runner.h"

namespace gml::gem::controller {

using ::gml::internal::api::core::v1::ExecutionSpec;
using ::gml::internal::api::core::v1::ModelSpec;
using ::gml::internal::controlplane::egw::v1::BridgeRequest;
using ::gml::internal::controlplane::egw::v1::BridgeResponse;

class ModelExecHandler : public MessageHandler {
 public:
  ModelExecHandler() = delete;
  ModelExecHandler(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*,
                   exec::core::ControlExecutionContext*);

  ~ModelExecHandler() override = default;

  Status HandleMessage(const BridgeResponse& msg) override;

  Status Init() override;
  Status Finish() override;

 private:
  class RunModelTask;
  void HandleRunModelFinished();

  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
  event::RunnableAsyncTaskUPtr running_task_ = nullptr;
  std::atomic<bool> stop_signal_ = false;
  ExecutionSpec exec_spec_;
  ModelSpec model_spec_;
};

}  // namespace gml::gem::controller
