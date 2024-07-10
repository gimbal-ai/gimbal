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

#include "src/gem/fakegem/controller.h"

#include <chrono>
#include <random>
#include <thread>

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

DEFINE_string(max_init_delay_seconds, gflags::StringFromEnv("GML_MAX_INIT_DELAY_S", "30"),
              "The maximum number of seconds of random delay to inject into the initialization.");

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

  // Inject a random amount of latency so that a fleet of fake GEMs have staggered timeseries data.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, std::stoi(FLAGS_max_init_delay_seconds));
  int random_delay_seconds = distrib(gen);
  LOG(INFO) << "Delaying video stream start by " << random_delay_seconds << " seconds.";
  std::this_thread::sleep_for(std::chrono::seconds(random_delay_seconds));

  // Register message handlers.
  // Use the same handlers as real GEM for heartbeat and info
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
