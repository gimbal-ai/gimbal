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

#include "src/gem/core/runner.h"
#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace core {

Status Runner::Init(const std::map<std::string, mediapipe::Packet>& extra_side_packets) {
  GML_RETURN_IF_ERROR(graph_.Initialize(spec_.graph()));

  side_packets_.insert(extra_side_packets.begin(), extra_side_packets.end());

  for (const auto& required_ctx : spec_.required_execution_context()) {
    GML_ASSIGN_OR_RETURN(auto ctx,
                         plugin_registry_->BuildExecutionContext(required_ctx.name(), spec_));
    auto packet = mediapipe::MakePacket<ExecutionContext*>(ctx.get());
    side_packets_.emplace(required_ctx.input_side_packet_name(), std::move(packet));
    exec_ctxs_.emplace_back(std::move(ctx));
  }

  initialized_ = true;
  return Status::OK();
}

Status Runner::Start() {
  GML_RETURN_IF_ERROR(graph_.StartRun(side_packets_));
  started_ = true;
  return Status::OK();
}

Status Runner::Stop() {
  graph_.Cancel();
  return Wait();
}

Status Runner::Wait() {
  GML_RETURN_IF_ERROR(graph_.WaitUntilDone());
  return Status::OK();
}

}  // namespace core
}  // namespace gem
}  // namespace gml