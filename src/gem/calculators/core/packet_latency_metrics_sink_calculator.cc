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

#include "src/gem/calculators/core/packet_latency_metrics_sink_calculator.h"

#include <vector>

#include <mediapipe/calculators/util/latency.pb.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_registry.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/metrics_utils.h"
#include "src/gem/calculators/core/optionspb/packet_latency_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kFinishedTag = "FINISHED";

absl::Status PacketLatencyMetricsSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<mediapipe::PacketLatency>();
  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Set<bool>();
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status PacketLatencyMetricsSinkCalculator::Open(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::PacketLatencyMetricsSinkCalculatorOptions>();
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  latency_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      absl::Substitute("gml_gem_$0_latency_seconds", options.name()), "Packet latency in seconds.",
      metrics_utils::kLatencySecondsBucketBounds);
  return absl::OkStatus();
}

absl::Status PacketLatencyMetricsSinkCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::PacketLatencyMetricsSinkCalculatorOptions>();
  auto& packet_latency = cc->Inputs().Index(0).Get<mediapipe::PacketLatency>();
  latency_hist_->Record(
      static_cast<double>(packet_latency.current_latency_usec()) / 1000.0 / 1000.0,
      options.metric_attributes());

  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs()
        .Tag(kFinishedTag)
        .AddPacket(mediapipe::MakePacket<bool>(true).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status PacketLatencyMetricsSinkCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(PacketLatencyMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
