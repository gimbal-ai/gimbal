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

const std::vector<double> kLatencyBucketBounds = {0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
                                                  0.035, 0.040, 0.045, 0.050, 0.075, 0.100, 0.150};

Status ClockLatencyMetricsSinkCalculator::BuildMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::ClockLatencyMetricsSinkCalculatorOptions>();
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  latency_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      absl::Substitute("gml_gem_$0_latency_seconds", options.name()), "Packet latency in seconds.",
      kLatencyBucketBounds);
  return Status::OK();
}

Status ClockLatencyMetricsSinkCalculator::RecordMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::ClockLatencyMetricsSinkCalculatorOptions>();

  uint64_t max_latency_usecs = 0;
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    if (cc->Inputs().Index(i).IsEmpty()) {
      continue;
    }

    cc->Inputs().Index(i).Get<absl::Duration>();
    auto& latency = cc->Inputs().Index(i).Get<absl::Duration>();
    uint64_t latency_usecs = ToChronoMicroseconds(latency).count();
    max_latency_usecs = std::max(max_latency_usecs, latency_usecs);
  }

  if (max_latency_usecs == 0) {
    return Status::OK();
  }

  auto latency_seconds = static_cast<double>(max_latency_usecs) / 1000.0 / 1000.0;
  latency_hist_->Record(latency_seconds, options.metric_attributes());
  return Status::OK();
}

REGISTER_CALCULATOR(ClockLatencyMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
