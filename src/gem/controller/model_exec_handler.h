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
#include "src/common/event/task.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/cached_blob_store.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/message_handler.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/exec/core/runner/runner.h"

namespace gml::gem::controller {

class ModelExecHandler : public MessageHandler {
 public:
  ModelExecHandler() = delete;
  ModelExecHandler(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*, CachedBlobStore*,
                   exec::core::ControlExecutionContext*);

  ~ModelExecHandler() override = default;

  Status HandleMessage(const ::gml::internal::controlplane::egw::v1::BridgeResponse& msg) override;
  Status HandlePhysicalPipelineSpecUpdate(
      const ::gml::internal::api::core::v1::PhysicalPipelineSpecUpdate&);

  Status Init() override;
  Status Finish() override;

 private:
  class RunModelTask;
  void HandleRunModelFinished(sole::uuid);
  void HandleModelStatusUpdate(sole::uuid, ::gml::internal::api::core::v1::PhysicalPipelineStatus*);
  Status GetDefaultVideoExecutionGraph(internal::api::core::v1::PhysicalPipelineSpecUpdate*);

  CachedBlobStore* blob_store_;

  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
  event::RunnableAsyncTaskUPtr running_task_ = nullptr;
  sole::uuid physical_pipeline_id_;
  absl::base_internal::SpinLock exec_graph_lock_;
  std::unique_ptr<internal::api::core::v1::PhysicalPipelineSpecUpdate> queued_execution_graph_
      ABSL_GUARDED_BY(exec_graph_lock_);
  std::atomic<bool> stop_signal_ = false;
};

}  // namespace gml::gem::controller
