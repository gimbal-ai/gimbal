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
