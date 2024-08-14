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

#include "src/gem/calculators/core/clock_latency_metrics_sink_calculator.h"

#include <vector>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/clock_latency_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kFinishedTag = "FINISHED";

const std::vector<double> kLatencyBucketBounds = {0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
                                                  0.035, 0.040, 0.045, 0.050, 0.075, 0.100, 0.150};

absl::Status ClockLatencyMetricsSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  for (mediapipe::CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
    cc->Inputs().Get(id).Set<absl::Duration>();
  }
  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Set<bool>();
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ClockLatencyMetricsSinkCalculator::Open(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::ClockLatencyMetricsSinkCalculatorOptions>();
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  latency_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      absl::Substitute("gml_gem_$0_latency_seconds", options.name()), "Packet latency in seconds.",
      kLatencyBucketBounds);
  return absl::OkStatus();
}

absl::Status ClockLatencyMetricsSinkCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::ClockLatencyMetricsSinkCalculatorOptions>();

  // We set the max latency to 0 because we can get negative latencies. This happens when
  // the node we measure is too fast so that the ClockTimestampCalculators are scheduled in
  // the incorrect order.
  // This bug is filed in GML-1320.
  int64_t max_latency_usecs = 0;
  // We ensure that we only record metrics if there is at least one input.
  bool has_input = false;
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    if (cc->Inputs().Index(i).IsEmpty()) {
      continue;
    }
    has_input = true;

    cc->Inputs().Index(i).Get<absl::Duration>();
    auto& latency = cc->Inputs().Index(i).Get<absl::Duration>();
    int64_t latency_usecs = absl::ToInt64Microseconds(latency);
    max_latency_usecs = std::max(max_latency_usecs, latency_usecs);
  }

  if (!has_input) {
    return absl::OkStatus();
  }

  auto latency_seconds = static_cast<double>(max_latency_usecs) / 1000.0 / 1000.0;
  latency_hist_->Record(latency_seconds, options.metric_attributes());

  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs()
        .Tag(kFinishedTag)
        .AddPacket(mediapipe::MakePacket<bool>(true).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status ClockLatencyMetricsSinkCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ClockLatencyMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
