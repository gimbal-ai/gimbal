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

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/flow_limiter_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

/**
 * FlowLimiterMetricsSinkCalculator Graph API:
 *
 *  Inputs:
 *    bool Allow signal from flow limiter indicating drop/allow status for each packet.
 *
 *  No outputs, outputs stats to opentelemetry.
 *
 */
class FlowLimiterMetricsSinkCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<bool>();
    return absl::OkStatus();
  }

  absl::Status Open(mediapipe::CalculatorContext*) override {
    auto& metrics_system = metrics::MetricsSystem::GetInstance();

    allow_counter_ = metrics_system.GetOrCreateCounter(
        "gml_gem_pipe_flow_limiter_allows",
        "Number of packets allowed to pass through the flow limiter.");

    drop_counter_ = metrics_system.GetOrCreateCounter(
        "gml_gem_pipe_flow_limiter_drops", "Number of packets dropped by the flow limiter.");

    return absl::OkStatus();
  }

  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    const auto& options = cc->Options<optionspb::FlowLimiterMetricsSinkCalculatorOptions>();

    bool allowed = cc->Inputs().Index(0).Get<bool>();
    if (allowed) {
      allow_counter_->Add(1, options.metric_attributes());
    } else {
      drop_counter_->Add(1, options.metric_attributes());
    }

    return absl::OkStatus();
  }

 private:
  opentelemetry::metrics::Counter<uint64_t>* allow_counter_;
  opentelemetry::metrics::Counter<uint64_t>* drop_counter_;
};

REGISTER_CALCULATOR(FlowLimiterMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
