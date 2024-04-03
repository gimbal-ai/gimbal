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

#include "src/gem/controller/model_exec_handler.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/event/dispatcher.h"
#include "src/common/event/task.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/calculators/plugin/argus/optionspb/argus_cam_calculator_options.pb.h"
#include "src/gem/calculators/plugin/opencv_cam/optionspb/opencv_cam_calculator_options.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/exec/core/runner/runner.h"
#include "src/gem/plugins/registry.h"

DEFINE_string(default_opencv_pbtxt, "src/gem/static/default_opencv_graph.pbtxt",
              "Path to default opencv video stream execution graph in pbtxt format");
DEFINE_string(default_argus_pbtxt, "src/gem/static/default_argus_graph.pbtxt",
              "Path to default argus video stream execution graph in pbtxt format");

namespace gml::gem::controller {

using ::gml::gem::exec::core::ExecutionContext;
using ::gml::gem::exec::core::Model;
using ::gml::internal::api::core::v1::ApplyExecutionGraph;
using ::gml::internal::api::core::v1::DeleteExecutionGraph;
using gml::internal::api::core::v1::EDGE_CP_TOPIC_EXEC;
using gml::internal::api::core::v1::ExecutionGraphState;
using gml::internal::api::core::v1::ExecutionGraphStatus;
using gml::internal::api::core::v1::ExecutionGraphStatusUpdate;
using ::gml::internal::api::core::v1::ExecutionSpec;
using ::gml::internal::controlplane::egw::v1::BridgeResponse;

namespace {
Status LoadPbtxt(const std::string& path, google::protobuf::Message* msg) {
  std::ifstream f(bazel::RunfilePath(std::filesystem::path(path)));
  std::stringstream buf;
  buf << f.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(buf.str(), msg)) {
    return error::InvalidArgument("Failed to parse spec from pbtxt");
  }
  return Status::OK();
}
}  // namespace

ModelExecHandler::ModelExecHandler(gml::event::Dispatcher* dispatcher, GEMInfo* info,
                                   GRPCBridge* bridge, CachedBlobStore* blob_store,
                                   exec::core::ControlExecutionContext* ctrl_exec_ctx)
    : MessageHandler(dispatcher, info, bridge),
      blob_store_(blob_store),
      ctrl_exec_ctx_(ctrl_exec_ctx) {}

class ModelExecHandler::RunModelTask : public event::AsyncTask {
 public:
  RunModelTask(ModelExecHandler* parent, ExecutionSpec exec_spec,
               exec::core::ControlExecutionContext* ctrl_exec_ctx, sole::uuid physical_pipeline_id,
               int64_t version)
      : parent_(parent),
        exec_spec_(std::move(exec_spec)),
        ctrl_exec_ctx_(ctrl_exec_ctx),
        physical_pipeline_id_(physical_pipeline_id),
        version_(version) {}

  Status PreparePluginExecutionContexts() {
    SendStatusUpdate(ExecutionGraphState::EXECUTION_GRAPH_STATE_COMPILING, "");

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

    SendStatusUpdate(ExecutionGraphState::EXECUTION_GRAPH_STATE_READY, "");
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

    SendStatusUpdate(ExecutionGraphState::EXECUTION_GRAPH_STATE_DEPLOYED, "");
    GML_RETURN_IF_ERROR(runner.Start());
    GML_RETURN_IF_ERROR(
        ::gml::metrics::MetricsSystem::GetInstance().RegisterAuxMetricsProvider(&runner));
    while (!parent_->stop_signal_.load() && !runner.HasError()) {
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
      SendStatusUpdate(ExecutionGraphState::EXECUTION_GRAPH_STATE_FAILED, "");
      LOG(ERROR) << "Failed to run model: " << physical_pipeline_id_ << " " << s.msg();
    }
  }

  void Done() override {
    SendStatusUpdate(ExecutionGraphState::EXECUTION_GRAPH_STATE_TERMINATING, "");

    parent_->HandleRunModelFinished(physical_pipeline_id_);
  }

  void SendStatusUpdate(ExecutionGraphState state, std::string reason) {
    ExecutionGraphStatus status;
    status.set_state(state);
    status.set_version(version_);
    status.set_reason(reason);
    parent_->HandleModelStatusUpdate(physical_pipeline_id_, &status);
  }

 private:
  ModelExecHandler* parent_;
  ExecutionSpec exec_spec_;

  exec::core::ControlExecutionContext* ctrl_exec_ctx_;
  std::unique_ptr<ExecutionContext> cpu_exec_ctx_;
  std::vector<std::unique_ptr<Model>> models_;
  absl::flat_hash_map<std::string, std::unique_ptr<ExecutionContext>> model_exec_ctxs_;
  sole::uuid physical_pipeline_id_;
  int64_t version_;
};

Status ModelExecHandler::HandleMessage(const BridgeResponse& msg) {
  ApplyExecutionGraph eg;
  if (msg.msg().UnpackTo(&eg)) {
    return this->HandleApplyExecutionGraph(eg);
  }

  DeleteExecutionGraph dg;
  if (msg.msg().UnpackTo(&dg)) {
    return this->HandleDeleteExecutionGraph(dg);
  }

  LOG(ERROR) << "Failed to unpack message. Received message of type: " << msg.msg().type_url()
             << " . Ignoring...";

  return Status::OK();
}

Status ModelExecHandler::HandleDeleteExecutionGraph(const DeleteExecutionGraph& dg) {
  absl::base_internal::SpinLockHolder lock(&exec_graph_lock_);

  auto id = ParseUUID(dg.physical_pipeline_id());

  // If there's a running task and its ID matches the one to delete, stop it.
  if (running_task_ != nullptr && id == physical_pipeline_id_) {
    LOG(INFO) << "Model is running... Signaling the running task to stop";
    stop_signal_.store(true);

    // If there are no queued execution graphs, run the default pipeline.
    if (queued_execution_graph_ == nullptr) {
      LOG(INFO) << "No other models are queued... Starting the DefaultVideoExecutionGraph";
      ApplyExecutionGraph eg;
      GML_RETURN_IF_ERROR(this->GetDefaultVideoExecutionGraph(&eg));
      return this->HandleApplyExecutionGraph(eg);
    }

    return Status::OK();
  }

  // If the queued execution graph's ID matches the one to delete, remove it from the queue.
  if (queued_execution_graph_ != nullptr &&
      id == ParseUUID(queued_execution_graph_->physical_pipeline_id())) {
    LOG(INFO) << "Model is running... Signaling the running task to stop";
    queued_execution_graph_ = nullptr;

    return Status::OK();
  }

  LOG(INFO) << "Could not find model to delete from device  " << id;

  return Status::OK();
}

Status ModelExecHandler::HandleApplyExecutionGraph(const ApplyExecutionGraph& eg) {
  absl::base_internal::SpinLockHolder lock(&exec_graph_lock_);

  auto id = ParseUUID(eg.physical_pipeline_id());

  if (running_task_ != nullptr) {
    LOG(INFO) << "Model already running... Queuing up RunModel Request";
    queued_execution_graph_ = std::make_unique<ApplyExecutionGraph>(eg);
    stop_signal_.store(true);
    return Status::OK();
  }

  LOG(INFO) << "Starting model execution  " << id;
  queued_execution_graph_ = nullptr;
  ctrl_exec_ctx_->SetLogicalPipelineID(ParseUUID(eg.logical_pipeline_id()));

  if (!eg.has_spec()) {
    LOG(ERROR) << "Missing spec in ApplyExecutionGraph msg";
    return Status::OK();
  }

  stop_signal_.store(false);

  auto task = std::make_unique<RunModelTask>(this, eg.spec().graph(), ctrl_exec_ctx_, id,
                                             eg.spec().version());

  running_task_ = dispatcher()->CreateAsyncTask(std::move(task));
  running_task_->Run();
  physical_pipeline_id_ = id;

  return Status::OK();
}

Status ModelExecHandler::GetDefaultVideoExecutionGraph(ApplyExecutionGraph* eg) {
  auto caps = capabilities::core::DeviceCapabilities();
  auto& plugin_registry = plugins::Registry::GetInstance();
  for (auto& name : plugin_registry.RegisteredCapabilityListers()) {
    GML_ASSIGN_OR_RETURN(auto builder, plugin_registry.BuildCapabilityLister(name));
    auto s = builder->Populate(&caps);
    if (!s.ok()) {
      continue;
    }
  }
  // Choose the argus camera, or last camera in the array.
  gml::internal::api::core::v1::DeviceCapabilities_CameraInfo camera;
  for (auto& c : caps.cameras()) {
    camera = c;
    if (c.driver() ==
        gml::internal::api::core::v1::DeviceCapabilities_CameraInfo::CAMERA_DRIVER_ARGUS) {
      break;
    }
  }

  // Pass the camera options to the exec graph.
  auto spec = eg->mutable_spec()->mutable_graph();
  google::protobuf::Any any;
  if (camera.driver() ==
      gml::internal::api::core::v1::DeviceCapabilities_CameraInfo::CAMERA_DRIVER_ARGUS) {
    GML_RETURN_IF_ERROR(LoadPbtxt(FLAGS_default_argus_pbtxt, spec));
    gml::gem::calculators::argus::optionspb::ArgusCamSourceSubgraphOptions opts;
    opts.set_device_uuid(camera.camera_id());
    opts.set_target_frame_rate(30);
    any.PackFrom(opts);
  } else {
    GML_RETURN_IF_ERROR(LoadPbtxt(FLAGS_default_opencv_pbtxt, spec));
    gml::gem::calculators::opencv_cam::optionspb::OpenCVCamSourceSubgraphOptions opts;
    opts.set_device_filename(camera.camera_id());
    any.PackFrom(opts);
  }
  (*spec->mutable_graph()->mutable_node(0)->add_node_options()) = any;

  return Status::OK();
}

Status ModelExecHandler::Init() {
  // When the GEM starts up, we want to run the video stream by default. This finds the
  // device's capabilities and initializes the default video stream.
  ApplyExecutionGraph eg;

  GML_RETURN_IF_ERROR(this->GetDefaultVideoExecutionGraph(&eg));

  return this->HandleApplyExecutionGraph(eg);
}

Status ModelExecHandler::Finish() {
  if (running_task_ != nullptr) {
    stop_signal_.store(true);
  }
  return Status::OK();
}

void ModelExecHandler::HandleRunModelFinished(sole::uuid physical_pipeline_id) {
  LOG(INFO) << "Model execution finished " << physical_pipeline_id;
  dispatcher()->DeferredDelete(std::move(running_task_));
  running_task_ = nullptr;
  physical_pipeline_id_ = sole::uuid{};

  absl::base_internal::SpinLockHolder lock(&exec_graph_lock_);
  if (queued_execution_graph_ != nullptr) {
    auto eg = *queued_execution_graph_.get();
    auto post_cb = [this, eg]() mutable { ECHECK_OK(this->HandleApplyExecutionGraph(eg)); };

    dispatcher()->Post(event::PostCB(std::move(post_cb)));
  }
}

void ModelExecHandler::HandleModelStatusUpdate(sole::uuid physical_pipeline_id,
                                               ExecutionGraphStatus* status) {
  ExecutionGraphStatusUpdate update;

  // Don't send the message if it is from the "default" pipeline.
  if (physical_pipeline_id.ab == 0 && physical_pipeline_id.cd == 0) {
    return;
  }

  auto mutable_id = update.mutable_physical_pipeline_id();
  ToProto(physical_pipeline_id, mutable_id);
  (*update.mutable_status()) = *status;

  auto s = bridge()->SendMessageToBridge(EDGE_CP_TOPIC_EXEC, update);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to send state update";
  }
}

}  // namespace gml::gem::controller
