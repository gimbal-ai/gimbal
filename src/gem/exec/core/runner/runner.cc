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

#include "src/gem/exec/core/runner/runner.h"

#include "src/common/base/base.h"
#include "src/gem/exec/core/runner/utils/mp_to_otel_metrics.h"

namespace gml::gem::exec::core {

Status Runner::Init(const std::map<std::string, mediapipe::Packet>& extra_side_packets) {
  GML_RETURN_IF_ERROR(graph_.Initialize(spec_.graph()));

  side_packets_.insert(extra_side_packets.begin(), extra_side_packets.end());

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
  if (graph_.HasError()) {
    absl::Status graph_status;
    graph_.GetCombinedErrors(&graph_status);
    GML_RETURN_IF_ERROR(graph_status);
  }

  GML_RETURN_IF_ERROR(graph_.WaitUntilDone());

  absl::Status graph_status;
  graph_.GetCombinedErrors(&graph_status);
  GML_RETURN_IF_ERROR(graph_status);

  return Status::OK();
}

Status Runner::CollectMediaPipeMetrics(
    opentelemetry::proto::metrics::v1::ResourceMetrics* metrics) {
  std::vector<mediapipe::CalculatorProfile> profiles;
  GML_RETURN_IF_ERROR(
      StatusAdapter<absl::Status>(graph_.profiler()->GetCalculatorProfiles(&profiles)));
  GML_RETURN_IF_ERROR(
      utils::CalculatorProfileVecToOTelProto(profiles, start_time_unix_nanos_, metrics));

  return Status::OK();
}

}  // namespace gml::gem::exec::core
