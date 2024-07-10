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

#include <utility>

#include <mediapipe/framework/calculator_graph.h>

#include "src/api/corepb/v1/model_exec.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/plugins/registry.h"

namespace gml::gem::exec::core {

/**
 * Runner executes an ExecutionSpec until stopped.
 */
class Runner : public gml::metrics::AuxMetricsProvider {
 public:
  explicit Runner(::gml::internal::api::core::v1::ExecutionSpec spec) : spec_(std::move(spec)) {
    // Record the start time. This is used for metrics.
    auto now = std::chrono::system_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    start_time_unix_nanos_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();
  }

  Status Init(const std::map<std::string, mediapipe::Packet>& extra_side_packets);
  Status Start();
  Status Stop();
  Status Wait();

  bool HasError() { return graph_.HasError(); }

  /**
   * Returns performance information about the running graph. Adds results to the metrics proto.
   */
  Status CollectMediaPipeMetrics(opentelemetry::proto::metrics::v1::ResourceMetrics* metrics);

  Status CollectMetrics(opentelemetry::proto::metrics::v1::ResourceMetrics* metrics) override {
    GML_RETURN_IF_ERROR(CollectMediaPipeMetrics(metrics));
    return Status::OK();
  }

  template <typename TPacket>
  Status AddOutputStreamCallback(
      const std::string& stream_name,
      std::function<Status(const TPacket&, const mediapipe::Timestamp&)> cb) {
    if (!initialized_) {
      return {types::CODE_INVALID_ARGUMENT,
              "Cannot add output stream callback before initialization"};
    }
    if (started_) {
      return {types::CODE_INVALID_ARGUMENT,
              "Cannot add output stream callback after graph is already running"};
    }

    GML_RETURN_IF_ERROR(graph_.ObserveOutputStream(
        stream_name, [&](const mediapipe::Packet& packet) -> absl::Status {
          const auto& val = packet.Get<TPacket>();
          const auto& ts = packet.Timestamp();
          GML_ABSL_RETURN_IF_ERROR(cb(val, ts));
          return absl::OkStatus();
        }));
    return Status::OK();
  }

 private:
  ::gml::internal::api::core::v1::ExecutionSpec spec_;
  mediapipe::CalculatorGraph graph_;
  std::map<std::string, mediapipe::Packet> side_packets_;
  bool initialized_ = false;
  bool started_ = false;
  int64_t start_time_unix_nanos_ = 0;
};

}  // namespace gml::gem::exec::core
