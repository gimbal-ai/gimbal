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
  std::string graph_proto_text;
  google::protobuf::TextFormat::PrintToString(graph_.Config(), &graph_proto_text);
  VLOG(1) << "Graph:\n" << graph_proto_text;

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
