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

#include "src/gem/controller/model_exec_handler.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

#include <absl/strings/str_replace.h>
#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/event/dispatcher.h"
#include "src/common/event/task.h"
#include "src/common/mediapipe/sanitize.h"
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
using gml::internal::api::core::v1::EDGE_CP_TOPIC_EXEC;
using ::gml::internal::api::core::v1::ExecutionSpec;
using ::gml::internal::api::core::v1::PhysicalPipelineSpecUpdate;
using ::gml::internal::api::core::v1::PhysicalPipelineStatus;
using gml::internal::api::core::v1::PhysicalPipelineStatusUpdate;
using gml::internal::api::core::v1::PipelineState;
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
               std::string device_hash, int64_t version)
      : parent_(parent),
        exec_spec_(std::move(exec_spec)),
        ctrl_exec_ctx_(ctrl_exec_ctx),
        physical_pipeline_id_(physical_pipeline_id),
        device_resource_hash_(std::move(device_hash)),
        version_(version) {}

  Status PreparePluginExecutionContexts() {
    SendStatusUpdate(PipelineState::PIPELINE_STATE_PENDING, "");

    auto& plugin_registry = plugins::Registry::GetInstance();

    // We use a shared CPU context for all nodes in the mediapipe execution graph.
    GML_ASSIGN_OR_RETURN(cpu_exec_ctx_,
                         plugin_registry.BuildExecutionContext("cpu_tensor", nullptr));

    // Each model needs its own model execution context.
    for (const auto& [i, model_spec] : Enumerate(exec_spec_.model_spec())) {
      std::string plugin = model_spec.runtime();
      runtime_ = plugin;
      std::string context_name = absl::StrCat(model_spec.name(), "_exec_ctx");
      // mediapipe only accepts lowercase characters and "_".
      context_name = gml::mp::SanitizeNameForMediapipeGraph(context_name);

      for (const auto& asset : model_spec.named_asset()) {
        std::string name = model_spec.name();
        if (asset.name() != "") {
          name += ":" + asset.name();
        }
        LOG(INFO) << "Downloading model asset: " << name;
        GML_CHECK_OK(parent_->blob_store_->EnsureBlobExists(ParseUUID(asset.file().file_id()).str(),
                                                            asset.file().sha256_hash(),
                                                            asset.file().size_bytes()));
      }

      GML_ASSIGN_OR_RETURN(auto model,
                           plugin_registry.BuildModel(plugin, parent_->blob_store_, model_spec));
      models_.emplace_back(std::move(model));

      GML_ASSIGN_OR_RETURN(auto model_exec_ctx,
                           plugin_registry.BuildExecutionContext(plugin, models_[i].get()));
      GML_RETURN_IF_ERROR(
          EmplaceNewKey(&model_exec_ctxs_, context_name, std::move(model_exec_ctx)));
    }

    SendStatusUpdate(PipelineState::PIPELINE_STATE_READY, "");
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

    SendStatusUpdate(PipelineState::PIPELINE_STATE_RUNNING, "");
    GML_RETURN_IF_ERROR(runner.Start());
    GML_RETURN_IF_ERROR(
        ::gml::metrics::MetricsSystem::GetInstance().RegisterAuxMetricsProvider(&runner));
    while (!parent_->stop_signal_.load() && !runner.HasError()) {
      std::this_thread::sleep_for(std::chrono::seconds{1});
    }

    GML_RETURN_IF_ERROR(runner.Stop());
    return Status::OK();
  }

  void Work() override {
    auto s = Run();
    if (!s.ok() && !parent_->stop_signal_.load()) {
      // TODO(michelle): This is not the prettiest error message, we should consider grouping
      // potential errors into more readable messages.
      SendStatusUpdate(PipelineState::PIPELINE_STATE_FAILED, s.msg());
      failed_ = true;
      LOG(ERROR) << "Failed to run model: " << physical_pipeline_id_ << " " << s.msg();
    } else {
      LOG(INFO) << "Model execution finished successfully";
    }
  }

  void Done() override {
    if (!failed_) {
      SendStatusUpdate(PipelineState::PIPELINE_STATE_TERMINATED, "");
    }
    parent_->HandleRunModelFinished(physical_pipeline_id_);
  }

  void SendStatusUpdate(PipelineState state, std::string reason) {
    PhysicalPipelineStatus status;
    status.set_state(state);
    status.set_version(version_);
    status.set_reason(reason);
    status.set_runtime(runtime_);
    status.set_device_resource_hash(device_resource_hash_);

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
  bool failed_ = false;
  std::string device_resource_hash_;
  int64_t version_;
  std::string runtime_;
};

Status ModelExecHandler::HandleMessage(const BridgeResponse& msg) {
  PhysicalPipelineSpecUpdate update;
  if (msg.msg().UnpackTo(&update)) {
    return this->HandlePhysicalPipelineSpecUpdate(update);
  }

  LOG(ERROR) << "Failed to unpack message. Received message of type: " << msg.msg().type_url()
             << " . Ignoring...";

  return Status::OK();
}

Status ModelExecHandler::HandlePhysicalPipelineSpecUpdate(
    const PhysicalPipelineSpecUpdate& update) {
  absl::base_internal::SpinLockHolder lock(&exec_graph_lock_);

  PhysicalPipelineSpecUpdate graph = update;
  PhysicalPipelineSpecUpdate default_graph;

  if (update.spec().state() == PipelineState::PIPELINE_STATE_TERMINATED) {
    // We should run the default pipeline.
    GML_RETURN_IF_ERROR(this->GetDefaultVideoExecutionGraph(&default_graph));

    graph = default_graph;
  }

  auto id = ParseUUID(graph.physical_pipeline_id());

  if (running_task_ != nullptr) {
    LOG(INFO) << "Model already running... Queuing up RunModel Request";
    queued_execution_graph_ = std::make_unique<PhysicalPipelineSpecUpdate>(graph);
    stop_signal_.store(true);
    return Status::OK();
  }

  LOG(INFO) << "Starting model execution  " << id;
  queued_execution_graph_ = nullptr;

  if (!graph.has_spec()) {
    LOG(ERROR) << "Missing spec in PhysicalPipelineSpecUpdate msg";
    return Status::OK();
  }

  stop_signal_.store(false);

  auto task =
      std::make_unique<RunModelTask>(this, graph.spec().graph(), ctrl_exec_ctx_, id,
                                     graph.spec().device_resource_hash(), graph.spec().version());

  running_task_ = dispatcher()->CreateAsyncTask(std::move(task));
  running_task_->Run();
  physical_pipeline_id_ = id;

  return Status::OK();
}

Status ModelExecHandler::GetDefaultVideoExecutionGraph(PhysicalPipelineSpecUpdate* update) {
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
  auto spec = update->mutable_spec()->mutable_graph();
  google::protobuf::Any any;
  if (camera.driver() ==
      gml::internal::api::core::v1::DeviceCapabilities_CameraInfo::CAMERA_DRIVER_ARGUS) {
    GML_RETURN_IF_ERROR(LoadPbtxt(FLAGS_default_argus_pbtxt, spec));
    gml::gem::calculators::argus::optionspb::ArgusCamSourceCalculatorOptions opts;
    opts.set_device_uuid(camera.camera_id());
    opts.set_target_frame_rate(30);
    any.PackFrom(opts);
  } else {
    GML_RETURN_IF_ERROR(LoadPbtxt(FLAGS_default_opencv_pbtxt, spec));
    gml::gem::calculators::opencv_cam::optionspb::OpenCVCamSourceCalculatorOptions opts;
    opts.set_device_filename(camera.camera_id());
    any.PackFrom(opts);
  }
  (*spec->mutable_graph()->mutable_graph_options(0)) = any;

  return Status::OK();
}

Status ModelExecHandler::Init() {
  // When the GEM starts up, we want to run the video stream by default. This finds the
  // device's capabilities and initializes the default video stream.
  PhysicalPipelineSpecUpdate update;

  GML_RETURN_IF_ERROR(this->GetDefaultVideoExecutionGraph(&update));

  return this->HandlePhysicalPipelineSpecUpdate(update);
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
    auto update = *queued_execution_graph_.get();
    auto post_cb = [this, update]() mutable {
      ECHECK_OK(this->HandlePhysicalPipelineSpecUpdate(update));
    };

    dispatcher()->Post(event::PostCB(std::move(post_cb)));
  }
}

void ModelExecHandler::HandleModelStatusUpdate(sole::uuid physical_pipeline_id,
                                               PhysicalPipelineStatus* status) {
  PhysicalPipelineStatusUpdate update;

  auto mutable_id = update.mutable_physical_pipeline_id();
  ToProto(physical_pipeline_id, mutable_id);
  (*update.mutable_status()) = *status;

  auto s = bridge()->SendMessageToBridge(EDGE_CP_TOPIC_EXEC, update);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to send state update";
  }
}

}  // namespace gml::gem::controller
