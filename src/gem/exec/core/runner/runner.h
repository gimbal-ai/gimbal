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

#include <mediapipe/framework/calculator_graph.h>

#include "src/common/base/base.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/plugins/registry.h"
#include "src/gem/specpb/execution.pb.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {

/**
 * Runner executes an ExecutionSpec until stopped.
 */
class Runner {
 public:
  explicit Runner(const specpb::ExecutionSpec& spec) : spec_(spec) {}

  Status Init(const std::map<std::string, mediapipe::Packet>& extra_side_packets);
  Status Start();
  Status Stop();
  Status Wait();

  bool HasError() { return graph_.HasError(); }

  /**
   * Returns performance information about the running graph.
   */
  Status GetCalculatorProfiles(std::vector<mediapipe::CalculatorProfile>* profiles);

  template <typename TPacket>
  Status AddOutputStreamCallback(
      const std::string& stream_name,
      std::function<Status(const TPacket&, const mediapipe::Timestamp&)> cb) {
    if (!initialized_) {
      return Status(types::CODE_INVALID_ARGUMENT,
                    "Cannot add output stream callback before initialization");
    }
    if (started_) {
      return Status(types::CODE_INVALID_ARGUMENT,
                    "Cannot add output stream callback after graph is already running");
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
  specpb::ExecutionSpec spec_;
  mediapipe::CalculatorGraph graph_;
  std::vector<std::unique_ptr<ExecutionContext>> exec_ctxs_;
  std::map<std::string, mediapipe::Packet> side_packets_;
  bool initialized_ = false;
  bool started_ = false;
};

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
