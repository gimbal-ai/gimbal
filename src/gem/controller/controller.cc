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
#include <chrono>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"
#include "src/common/system/mac_address.h"
#include "src/common/uuid/uuid.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.pb.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/controller/heartbeat.h"
#include "src/gem/controller/model_exec_handler.h"
#include "src/gem/controller/video_stream_handler.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

using gml::internal::api::core::v1::CPEdgeTopic;
using gml::internal::api::core::v1::EdgeCPMessage;
using gml::internal::api::core::v1::EdgeHeartbeat;
using gml::internal::api::core::v1::EdgeHeartbeatAck;

using gml::internal::controlplane::egw::v1::BridgeRequest;
using gml::internal::controlplane::egw::v1::BridgeResponse;
using gml::internal::controlplane::egw::v1::EGWService;
using gml::internal::controlplane::fleetmgr::v1::FleetMgrEdgeService;
using gml::internal::controlplane::fleetmgr::v1::RegisterRequest;
using gml::internal::controlplane::fleetmgr::v1::RegisterResponse;

using internal::api::core::v1::CP_EDGE_TOPIC_EXEC;
using internal::api::core::v1::CP_EDGE_TOPIC_STATUS;
using internal::api::core::v1::CP_EDGE_TOPIC_VIDEO;
using internal::api::core::v1::EDGE_CP_TOPIC_EXEC;
using internal::api::core::v1::EDGE_CP_TOPIC_STATUS;

DEFINE_string(device_serial, gflags::StringFromEnv("GML_DEVICE_SERIAL", ""),
              "Force set the serial number / ID for the device. Note this needs to be unique "
              "across devices.");

namespace {

constexpr size_t kMaxHostnameSize = 128;
// TODO(zasgar): Move to system.
StatusOr<std::string> GetHostname() {
  char hostname[kMaxHostnameSize];
  int err = gethostname(hostname, sizeof(hostname));
  if (err != 0) {
    return error::Unknown("Failed to get hostname");
  }
  return std::string(hostname);
}

// The device serial string. We selectively use the passed in flag, or
// the mac_address. If a unique_id cannot be found, we will currently error out.
StatusOr<std::string> GetDeviceSerial() {
  if (!FLAGS_device_serial.empty()) {
    return FLAGS_device_serial;
  }

  GML_ASSIGN_OR_RETURN(auto dev_reader, system::NetDeviceReader::Create());
  GML_ASSIGN_OR_RETURN(system::NetDevice dev, dev_reader->SystemMacAddress());
  if (!dev.mac_address().GloballyUnique()) {
    return error::FailedPrecondition("Failed to get unique mac address");
  }
  return dev.mac_address().str();
}

}  // namespace

Status Controller::Init() {
  GML_ASSIGN_OR_RETURN(info_.hostname, GetHostname());
  info_.pid = getpid();

  auto channel_creds = grpc::SslCredentials(grpc::SslCredentialsOptions());
  cp_chan_ = grpc::CreateChannel(cp_addr_, channel_creds);
  fmstub_ = std::make_unique<FleetMgrEdgeService::Stub>(cp_chan_);
  egwstub_ = std::make_unique<EGWService::Stub>(cp_chan_);

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

  RegisterResponse resp;
  auto s = fmstub_->Register(&ctx, req, &resp);
  if (!s.ok()) {
    return error::Internal(s.error_message());
  }

  GML_ASSIGN_OR_RETURN(info_.id, ParseUUID(resp.device_id()));
  LOG(INFO) << "Device ID: " << info_.id;
  LOG(INFO) << "Starting GRPC Bridge";

  bridge_ = std::make_unique<GRPCBridge>(cp_chan_, deploy_key_, info_.id);
  GML_RETURN_IF_ERROR(bridge_->Init());

  bridge_->RegisterOnMessageReadHandler(
      std::bind(&Controller::HandleMessage, this, std::placeholders::_1));

  ctrl_exec_ctx_ = std::make_unique<exec::core::ControlExecutionContext>();

  // Register message handlers.
  auto hb_handler = std::make_shared<HeartbeatHandler>(dispatcher(), &info_, bridge_.get());
  auto exec_handler =
      std::make_shared<ModelExecHandler>(dispatcher(), &info_, bridge_.get(), ctrl_exec_ctx_.get());
  auto video_handler = std::make_shared<VideoStreamHandler>(dispatcher(), &info_, bridge_.get(),
                                                            ctrl_exec_ctx_.get());

  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_STATUS, hb_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_EXEC, exec_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_VIDEO, video_handler));

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

  auto topic = msg->topic();
  auto it = message_handlers_.find(topic);
  if (it == message_handlers_.end()) {
    LOG(ERROR) << "Unhandled message topic: " << topic << " Message: " << msg->DebugString();
    return error::Unimplemented(absl::Substitute("no message handler for topic: $0", topic));
  }
  ECHECK_OK(it->second->HandleMessage(*msg)) << "message handler failed... for topic: " << topic
                                             << " ignoring. Message: " << msg->DebugString();
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
