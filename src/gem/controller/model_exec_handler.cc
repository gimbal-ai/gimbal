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

#include <unistd.h>

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/event/dispatcher.h"
#include "src/common/event/task.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/model_exec_handler.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/exec/core/runner/runner.h"
#include "src/gem/plugins/registry.h"

DEFINE_int32(frame_rate, 18, "Frame rate for encoding the video");

namespace gml::gem::controller {

using ::gml::gem::exec::core::ExecutionContext;
using ::gml::gem::exec::core::Model;
using ::gml::internal::api::core::v1::ApplyExecutionGraph;
using ::gml::internal::api::core::v1::ExecutionSpec;
using ::gml::internal::controlplane::egw::v1::BridgeResponse;

ModelExecHandler::ModelExecHandler(gml::event::Dispatcher* dispatcher, GEMInfo* info,
                                   GRPCBridge* bridge, CachedBlobStore* blob_store,
                                   exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(dispatcher, info, bridge),
      blob_store_(blob_store),
      ctrl_exec_ctx_(ctrl_exec_ctx) {}

class ModelExecHandler::RunModelTask : public event::AsyncTask {
 public:
  RunModelTask(ModelExecHandler* parent, ExecutionSpec exec_spec,
               exec::core::ControlExecutionContext* ctrl_exec_ctx)
      : parent_(parent), exec_spec_(std::move(exec_spec)), ctrl_exec_ctx_(ctrl_exec_ctx) {}

  Status PreparePluginExecutionContexts() {
    auto& plugin_registry = plugins::Registry::GetInstance();

    // We use a shared CPU context for all nodes in the mediapipe execution graph.
    GML_ASSIGN_OR_RETURN(cpu_exec_ctx_,
                         plugin_registry.BuildExecutionContext("cpu_tensor", nullptr));

    // Each model needs its own model execution context.
    for (const auto& [i, model_spec] : Enumerate(exec_spec_.model_spec())) {
      std::string plugin = model_spec.runtime();
      std::string context_name = absl::StrCat(model_spec.name(), "_", plugin, "_exec_ctx");

      GML_CHECK_OK(parent_->blob_store_->EnsureBlobExists(
          ParseUUID(model_spec.onnx_file().file_id()).str(), model_spec.onnx_file().sha256_hash(),
          model_spec.onnx_file().size_bytes()));

      GML_ASSIGN_OR_RETURN(auto model,
                           plugin_registry.BuildModel(plugin, parent_->blob_store_, model_spec));
      models_.emplace_back(std::move(model));

      GML_ASSIGN_OR_RETURN(auto model_exec_ctx,
                           plugin_registry.BuildExecutionContext(plugin, models_[i].get()));
      GML_RETURN_IF_ERROR(
          EmplaceNewKey(&model_exec_ctxs_, context_name, std::move(model_exec_ctx)));
    }

    return Status::OK();
  }

  Status Run() {
    GML_RETURN_IF_ERROR(PreparePluginExecutionContexts());

    std::map<std::string, mediapipe::Packet> side_packets;

    GML_RETURN_IF_ERROR(
        EmplaceNewKey(&side_packets, std::string("cpu_exec_ctx"),
                      mediapipe::MakePacket<ExecutionContext*>(cpu_exec_ctx_.get())));

    for (const auto& [context_name, model_exec_ctx] : model_exec_ctxs_) {
      GML_RETURN_IF_ERROR(
          EmplaceNewKey(&side_packets, context_name,
                        mediapipe::MakePacket<ExecutionContext*>(model_exec_ctx.get())));
    }

    GML_RETURN_IF_ERROR(EmplaceNewKey(&side_packets, std::string("ctrl_exec_ctx"),
                                      mediapipe::MakePacket<ExecutionContext*>(ctrl_exec_ctx_)));

    GML_RETURN_IF_ERROR(EmplaceNewKey(&side_packets, std::string("frame_rate"),
                                      mediapipe::MakePacket<int>(FLAGS_frame_rate)));

    exec::core::Runner runner(exec_spec_);
    GML_RETURN_IF_ERROR(runner.Init(side_packets));

    std::atomic_size_t num_frames = 0;
    std::atomic_size_t num_frames_dropped = 0;
    GML_RETURN_IF_ERROR(runner.AddOutputStreamCallback<bool>(
        "frame_allowed", [&](const bool& allowed, const mediapipe::Timestamp&) {
          num_frames++;
          if (!allowed) {
            num_frames_dropped++;
          }
          return Status::OK();
        }));

    GML_RETURN_IF_ERROR(runner.Start());

    while (!parent_->stop_signal_.load()) {
      auto dropped = num_frames_dropped.load();
      auto total = num_frames.load();
      if (total != 0) {
        LOG(INFO) << absl::Substitute(
            "Dropped $0/$1 frames ($2%)", dropped, total,
            static_cast<float>(100 * dropped) / static_cast<float>(total));
        num_frames.store(0);
        num_frames_dropped.store(0);
      }
      std::this_thread::sleep_for(std::chrono::seconds{5});
    }

    GML_RETURN_IF_ERROR(runner.Stop());
    return Status::OK();
  }

  void Work() override {
    auto s = Run();
    if (!s.ok()) {
      LOG(ERROR) << "Failed to run model: " << s.msg();
    }
  }

  void Done() override { parent_->HandleRunModelFinished(); }

 private:
  ModelExecHandler* parent_;
  ExecutionSpec exec_spec_;

  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
  std::unique_ptr<ExecutionContext> cpu_exec_ctx_;
  std::vector<std::unique_ptr<Model>> models_;
  absl::flat_hash_map<std::string, std::unique_ptr<ExecutionContext>> model_exec_ctxs_;
};

Status ModelExecHandler::HandleMessage(const BridgeResponse& msg) {
  ApplyExecutionGraph eg;
  if (!msg.msg().UnpackTo(&eg)) {
    LOG(ERROR) << "Failed to unpack apply execution graph message. Received message of type: "
               << msg.msg().type_url() << " . Ignoring...";
    return Status::OK();
  }

  if (running_task_ != nullptr) {
    LOG(INFO) << "Model already running skipping RunModel Request";
    return Status::OK();
  }

  LOG(INFO) << "Starting model execution";
  if (!eg.has_spec()) {
    LOG(ERROR) << "Missing spec in ApplyExecutionGraph msg";
    return Status::OK();
  }

  stop_signal_.store(false);

  auto task = std::make_unique<RunModelTask>(this, eg.spec().graph(), ctrl_exec_ctx_);

  running_task_ = dispatcher()->CreateAsyncTask(std::move(task));
  running_task_->Run();

  return Status::OK();
}

Status ModelExecHandler::Init() { return Status::OK(); }

Status ModelExecHandler::Finish() {
  if (running_task_ != nullptr) {
    stop_signal_.store(true);
  }
  return Status::OK();
}

void ModelExecHandler::HandleRunModelFinished() {
  LOG(INFO) << "Model Execution Finished";
  dispatcher()->DeferredDelete(std::move(running_task_));
  running_task_ = nullptr;
}

}  // namespace gml::gem::controller
