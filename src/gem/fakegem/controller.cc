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

#include "src/gem/fakegem/controller.h"

#include "src/common/uuid/uuid.h"
#include "src/gem/controller/controller.h"
#include "src/gem/controller/device_info.h"
#include "src/gem/controller/heartbeat.h"

namespace gml::gem::fakegem {

using internal::api::core::v1::CP_EDGE_TOPIC_EXEC;
using internal::api::core::v1::CP_EDGE_TOPIC_FILE_TRANSFER;
using internal::api::core::v1::CP_EDGE_TOPIC_INFO;
using internal::api::core::v1::CP_EDGE_TOPIC_METRICS;
using internal::api::core::v1::CP_EDGE_TOPIC_STATUS;
using internal::api::core::v1::CP_EDGE_TOPIC_VIDEO;

class FakeMessageHandlerWrapper : public controller::MessageHandler {
 public:
  FakeMessageHandlerWrapper(
      gml::event::Dispatcher* d, controller::GEMInfo* info, controller::GRPCBridge* b,
      std::function<Status(const internal::controlplane::egw::v1::BridgeResponse&)> cb)
      : MessageHandler(d, info, b), cb_(std::move(cb)) {}

  ~FakeMessageHandlerWrapper() override = default;

  Status HandleMessage(const internal::controlplane::egw::v1::BridgeResponse& msg) override {
    return cb_(msg);
  }

  Status Init() override { return Status::OK(); }
  Status Finish() override { return Status::OK(); }

 private:
  std::function<Status(const internal::controlplane::egw::v1::BridgeResponse&)> cb_;
};

Status FakeController::Init() {
  GML_RETURN_IF_ERROR(Register());

  fake_stream_writer_ = std::make_unique<StreamWriter>(bridge(), blob_store(), dispatcher());

  // Register message handlers.
  // Ue the same handlers as real GEM for heartbeat and info
  auto hb_handler = std::make_shared<controller::HeartbeatHandler>(dispatcher(), info(), bridge());

  // Use fake handlers for info, exec, video, and metrics.
  auto info_handler = std::make_shared<FakeMessageHandlerWrapper>(
      dispatcher(), info(), bridge(),
      [](const internal::controlplane::egw::v1::BridgeResponse&) { return Status::OK(); });

  auto exec_handler = std::make_shared<FakeMessageHandlerWrapper>(
      dispatcher(), info(), bridge(),
      [this](const internal::controlplane::egw::v1::BridgeResponse& msg) {
        internal::api::core::v1::ApplyExecutionGraph eg;
        if (!msg.msg().UnpackTo(&eg)) {
          LOG(ERROR)
              << "Failed to unpack apply execution graph message. Received message of type : "
              << msg.msg().type_url() << " . Ignoring...";
          return Status::OK();
        }

        auto id = ParseUUID(eg.physical_pipeline_id());
        LOG(INFO) << "Calling deploy pipeline with id: " << id.str();
        return fake_stream_writer_->StartModelStream(id);
      });
  auto video_handler = std::make_shared<FakeMessageHandlerWrapper>(
      dispatcher(), info(), bridge(),
      [this](const internal::controlplane::egw::v1::BridgeResponse&) {
        if (fake_stream_writer_->IsVideoStreamRunning()) {
          return Status::OK();
        }
        LOG(INFO) << "Starting video stream";
        return fake_stream_writer_->StartVideoStream();
      });

  auto metrics_handler = std::make_shared<FakeMessageHandlerWrapper>(
      dispatcher(), info(), bridge(),
      [](const internal::controlplane::egw::v1::BridgeResponse&) { return Status::OK(); });

  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_STATUS, hb_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_INFO, info_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_EXEC, exec_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_VIDEO, video_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_METRICS, metrics_handler));
  GML_CHECK_OK(RegisterMessageHandler(CP_EDGE_TOPIC_FILE_TRANSFER, file_downloader()));

  GML_RETURN_IF_ERROR(bridge()->Run());

  GML_RETURN_IF_ERROR(fake_stream_writer_->Run());

  hb_handler->EnableHeartbeats();
  return Status::OK();
}

Status FakeController::Stop(std::chrono::milliseconds timeout) {
  GML_RETURN_IF_ERROR(Controller::Stop(timeout));
  GML_RETURN_IF_ERROR(fake_stream_writer_->Stop());
  return Status::OK();
}

}  // namespace gml::gem::fakegem
