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

#include "src/gem/controller/controller.h"

#include <unistd.h>

#include <chrono>

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/system/hostname.h"
#include "src/common/system/version.h"
#include "src/common/uuid/uuid.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.pb.h"
#include "src/gem/controller/device_info.h"
#include "src/gem/controller/device_serial.h"
#include "src/gem/controller/file_downloader.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/controller/heartbeat.h"
#include "src/gem/controller/metrics_handler.h"
#include "src/gem/controller/model_exec_handler.h"
#include "src/gem/controller/system_metrics.h"
#include "src/gem/controller/video_stream_handler.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

using gml::internal::api::core::v1::CPEdgeTopic;

using gml::internal::controlplane::egw::v1::BridgeResponse;
using gml::internal::controlplane::egw::v1::EGWService;
using gml::internal::controlplane::fleetmgr::v1::FleetMgrEdgeService;
using gml::internal::controlplane::fleetmgr::v1::OSKind;
using gml::internal::controlplane::fleetmgr::v1::RegisterRequest;
using gml::internal::controlplane::fleetmgr::v1::RegisterResponse;

using internal::api::core::v1::CP_EDGE_TOPIC_EXEC;
using internal::api::core::v1::CP_EDGE_TOPIC_FILE_TRANSFER;
using internal::api::core::v1::CP_EDGE_TOPIC_INFO;
using internal::api::core::v1::CP_EDGE_TOPIC_METRICS;
using internal::api::core::v1::CP_EDGE_TOPIC_STATUS;
using internal::api::core::v1::CP_EDGE_TOPIC_VIDEO;

Status Controller::Register() {
  GML_ASSIGN_OR_RETURN(info_.hostname, system::GetHostname());
  info_.pid = getpid();

  auto channel_creds = grpc::SslCredentials(grpc::SslCredentialsOptions());
  cp_chan_ = grpc::CreateChannel(cp_addr_, channel_creds);
  fmstub_ = std::make_unique<FleetMgrEdgeService::Stub>(cp_chan_);
  egwstub_ = std::make_unique<EGWService::Stub>(cp_chan_);

  GML_ASSIGN_OR_RETURN(auto cpu_info_reader, gml::system::CPUInfoReader::Create());
  system_metrics_reader_ = std::make_unique<SystemMetricsReader>(
      &gml::metrics::MetricsSystem::GetInstance(), std::move(cpu_info_reader));

  // We need to make an RPC call to register the GEM and get the gem ID in return.
  // To get that we need to know the device serial number. We utilize the MAC address as
  // the serial number for the device.
  LOG(INFO) << "Trying to register GEM";
  RegisterRequest req;
  req.set_hostname(info_.hostname);
  auto serial_or_s = GetDeviceSerial();
  std::string selected_serial;
  if (!serial_or_s.ok()) {
    LOG(ERROR) << "Failed to get a unique identifier: " << serial_or_s.msg()
               << "\n manually specify the serial number with --device_serial";
    selected_serial = sole::uuid4().str();
  } else {
    selected_serial = serial_or_s.ConsumeValueOrDie();
  }
  req.set_device_serial(selected_serial);
  grpc::ClientContext ctx;
  ctx.AddMetadata("x-deploy-key", deploy_key_);

  auto* os_info = req.mutable_os();
  os_info->set_kind(OSKind::OS_KIND_LINUX);
  std::string os_version;
  GML_ASSIGN_OR_RETURN(os_version, system::GetUname());
  os_info->set_version(os_version);

  RegisterResponse resp;
  auto s = fmstub_->Register(&ctx, req, &resp);
  if (!s.ok()) {
    return error::Internal(s.error_message());
  }

  info_.id = ParseUUID(resp.device_id());
  LOG(INFO) << "Device ID: " << info_.id;
  LOG(INFO) << "Starting GRPC Bridge to: " << cp_addr_;

  bridge_ = std::make_unique<GRPCBridge>(cp_chan_, deploy_key_, info_.id);
  GML_RETURN_IF_ERROR(bridge_->Init());

  bridge_->RegisterOnMessageReadHandler(
      std::bind(&Controller::HandleMessage, this, std::placeholders::_1));

  ctrl_exec_ctx_ = std::make_unique<exec::core::ControlExecutionContext>();

  file_downloader_ = std::make_shared<FileDownloader>(dispatcher(), &info_, bridge_.get());
  GML_ASSIGN_OR_RETURN(blob_store_, CachedBlobStore::Create(file_downloader_));
  return Status::OK();
};

Status Controller::Init() {
  GML_RETURN_IF_ERROR(Register());
  // Register message handlers.
  auto hb_handler = std::make_shared<HeartbeatHandler>(dispatcher(), &info_, bridge_.get());
  auto info_handler = std::make_shared<DeviceInfoHandler>(dispatcher(), &info_, bridge_.get());
  auto exec_handler = std::make_shared<ModelExecHandler>(dispatcher(), &info_, bridge_.get(),
                                                         blob_store_.get(), ctrl_exec_ctx_.get());
  auto video_handler = std::make_shared<VideoStreamHandler>(dispatcher(), &info_, bridge_.get(),
                                                            ctrl_exec_ctx_.get());
  auto metrics_handler =
      std::make_shared<MetricsHandler>(dispatcher(), &info_, bridge_.get(), ctrl_exec_ctx_.get());

  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_STATUS, hb_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_INFO, info_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_EXEC, exec_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_VIDEO, video_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_METRICS, metrics_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_FILE_TRANSFER, file_downloader_));

  GML_RETURN_IF_ERROR(bridge_->Run());

  hb_handler->EnableHeartbeats();
  return Status::OK();
}

Status Controller::Run() {
  running_ = true;
  dispatcher_->Run(gml::event::Dispatcher::RunType::Block);
  running_ = false;
  return Status::OK();
}

Status Controller::Stop(std::chrono::milliseconds timeout) {
  GML_UNUSED(timeout);
  for (const auto& p : message_handlers_) {
    GML_RETURN_IF_ERROR(p.second->Finish());
  }
  return Status::OK();
}

Status Controller::HandleMessage(std::unique_ptr<BridgeResponse> msg) {
  VLOG(1) << "Handling message" << msg->DebugString();

  auto post_cb = [this, msg = std::move(msg)]() mutable {
    auto topic = msg->topic();
    auto it = message_handlers_.find(topic);
    if (it == message_handlers_.end()) {
      LOG(ERROR) << "Unhandled message topic: " << topic << " Message: " << msg->DebugString();
    }
    ECHECK_OK(it->second->HandleMessage(*msg)) << "message handler failed... for topic: " << topic
                                               << " ignoring. Message: " << msg->DebugString();
  };
  dispatcher()->Post(event::PostCB(std::move(post_cb)));

  return Status::OK();
}

Status Controller::RegisterMessageHandler(CPEdgeTopic topic,
                                          std::shared_ptr<MessageHandler> handler) {
  if (message_handlers_.contains(topic)) {
    return error::AlreadyExists("message handler already exists for case: $0", topic);
  }
  GML_RETURN_IF_ERROR(handler->Init());
  message_handlers_[topic] = std::move(handler);
  return Status::OK();
}

}  // namespace gml::gem::controller
