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
#include "src/gem/storage/fs_blob_store.h"

DEFINE_string(blob_store_dir, "/build/cache/", "Path to store blobs with the FilesystemBlobStore");
DEFINE_string(exec_spec_pbtxt, "/app/gem.runfiles/_main/src/gem/yolo_exec_spec.pbtxt",
              "Path to execution spec in pbtxt format");
DEFINE_int32(frame_rate, 18, "Frame rate for encoding the video");

namespace gml::gem::controller {

using ::gml::gem::exec::core::ExecutionContext;

namespace {
Status LoadPbtxt(const std::string& path, google::protobuf::Message* msg) {
  std::ifstream f(path);
  std::stringstream buf;
  buf << f.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(buf.str(), msg)) {
    return error::InvalidArgument("Failed to parse spec from pbtxt");
  }
  return Status::OK();
}
}  // namespace

ModelExecHandler::ModelExecHandler(gml::event::Dispatcher* dispatcher, GEMInfo* info,
                                   GRPCBridge* bridge,
                                   exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(dispatcher, info, bridge), ctrl_exec_ctx_(ctrl_exec_ctx) {}

class ModelExecHandler::RunModelTask : public event::AsyncTask {
 public:
  RunModelTask(ModelExecHandler* parent, exec::core::ControlExecutionContext* ctrl_exec_ctx)
      : parent_(parent), ctrl_exec_ctx_(ctrl_exec_ctx) {}

  Status Run() {
    auto& plugin_registry = plugins::Registry::GetInstance();

    GML_ASSIGN_OR_RETURN(auto store, storage::FilesystemBlobStore::Create(FLAGS_blob_store_dir));

    // TODO(oazizi): Support more than one model.
    if (parent_->exec_spec_.model_spec_size() != 1) {
      return error::Unimplemented("Currently only support a single model");
    }

    GML_ASSIGN_OR_RETURN(
        auto model,
        plugin_registry.BuildModel("tensorrt", store.get(), parent_->exec_spec_.model_spec()[0]));

    GML_ASSIGN_OR_RETURN(auto cpu_exec_ctx,
                         plugin_registry.BuildExecutionContext("cpu_tensor", nullptr));

    GML_ASSIGN_OR_RETURN(auto tensorrt_exec_ctx,
                         plugin_registry.BuildExecutionContext("tensorrt", model.get()));

    exec::core::Runner runner(parent_->exec_spec_);

    std::map<std::string, mediapipe::Packet> side_packets;
    side_packets.emplace("tensorrt_exec_ctx",
                         mediapipe::MakePacket<ExecutionContext*>(tensorrt_exec_ctx.get()));
    side_packets.emplace("cpu_exec_ctx",
                         mediapipe::MakePacket<ExecutionContext*>(cpu_exec_ctx.get()));
    side_packets.emplace("ctrl_exec_ctx", mediapipe::MakePacket<ExecutionContext*>(ctrl_exec_ctx_));
    side_packets.emplace("frame_rate", mediapipe::MakePacket<int>(FLAGS_frame_rate));

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
  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
};

Status ModelExecHandler::HandleMessage(const BridgeResponse& msg) {
  GML_UNUSED(msg);
  if (running_task_ != nullptr) {
    LOG(INFO) << "Model already running skipping RunModel Request";
    return Status::OK();
  }

  LOG(INFO) << "Starting model execution";

  stop_signal_.store(false);

  auto task = std::make_unique<RunModelTask>(this, ctrl_exec_ctx_);

  running_task_ = dispatcher()->CreateAsyncTask(std::move(task));
  running_task_->Run();

  return Status::OK();
}

Status ModelExecHandler::Init() {
  GML_RETURN_IF_ERROR(LoadPbtxt(FLAGS_exec_spec_pbtxt, &exec_spec_));
  return Status::OK();
}

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
