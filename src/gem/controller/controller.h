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

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/common/base/base.h"
#include "src/common/event/event.h"
#include "src/common/system/system.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/controlplane/fleetmgr/fmpb/v1/fmpb.grpc.pb.h"
#include "src/gem/controller/cached_blob_store.h"
#include "src/gem/controller/gem_metrics.h"
#include "src/gem/controller/message_handler.h"
#include "src/gem/controller/system_metrics.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::controller {

class ControllerBase : public gml::NotCopyable {
 public:
  virtual Status Stop(std::chrono::milliseconds timeout) = 0;
};

class Controller : public ControllerBase {
 public:
  Controller() = delete;
  Controller(std::string_view deploy_key, std::string_view cp_addr)
      : deploy_key_(std::string(deploy_key)),
        cp_addr_(std::string(cp_addr)),
        time_system_(std::make_unique<gml::event::RealTimeSystem>()),
        api_(std::make_unique<gml::event::APIImpl>(time_system_.get())),
        dispatcher_(api_->AllocateDispatcher("controller")),
        gem_metrics_reader_(
            std::make_unique<GEMMetricsReader>(&metrics::MetricsSystem::GetInstance())) {}

  virtual ~Controller() = default;

  Status Register();
  virtual Status Init();

  Status Run();

  Status Stop(std::chrono::milliseconds timeout) override;

 protected:
  gml::event::Dispatcher* dispatcher() { return dispatcher_.get(); }
  Status RegisterMessageHandler(gml::internal::api::core::v1::CPEdgeTopic,
                                std::shared_ptr<MessageHandler> handler);
  Status HandleMessage(std::unique_ptr<gml::internal::controlplane::egw::v1::BridgeResponse> msg);
  GEMInfo* info() { return &info_; }
  GRPCBridge* bridge() { return bridge_.get(); }
  std::shared_ptr<FileDownloader> file_downloader() { return file_downloader_; }
  CachedBlobStore* blob_store() { return blob_store_.get(); }

 private:
  // Marks if the controller is still running. Force stopping will cause un-graceful termination.
  std::atomic<bool> running_ = false;
  std::string deploy_key_;
  GEMInfo info_;
  std::string cp_addr_;
  // The time system to use (real or simulated).
  std::unique_ptr<gml::event::TimeSystem> time_system_;
  gml::event::APIUPtr api_;

  gml::event::DispatcherUPtr dispatcher_ = nullptr;
  std::unique_ptr<GRPCBridge> bridge_;
  std::unique_ptr<CachedBlobStore> blob_store_;
  std::shared_ptr<FileDownloader> file_downloader_;

  std::shared_ptr<grpc::Channel> cp_chan_;
  std::unique_ptr<gml::internal::controlplane::fleetmgr::v1::FleetMgrEdgeService::Stub> fmstub_;
  std::unique_ptr<gml::internal::controlplane::egw::v1::EGWService::Stub> egwstub_;
  absl::flat_hash_map<gml::internal::api::core::v1::CPEdgeTopic, std::shared_ptr<MessageHandler>>
      message_handlers_;

  std::unique_ptr<exec::core::ControlExecutionContext> ctrl_exec_ctx_;
  std::unique_ptr<SystemMetricsReader> system_metrics_reader_;
  std::unique_ptr<GEMMetricsReader> gem_metrics_reader_;
};

}  // namespace gml::gem::controller
