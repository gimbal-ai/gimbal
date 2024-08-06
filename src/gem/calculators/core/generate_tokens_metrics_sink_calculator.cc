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

#include "src/gem/calculators/core/generate_tokens_metrics_sink_calculator.h"

#include <absl/strings/str_cat.h>
#include <opentelemetry/metrics/provider.h>

#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/generate_tokens_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kOutputTokenIDsTag = "OUTPUT_TOKENS";
constexpr std::string_view kInputTokenIDsTag = "INPUT_TOKENS";
constexpr std::string_view kOutputEOSTag = "OUTPUT_EOS";

absl::Status GenerateTokensMetricsSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kInputTokenIDsTag).Set<std::vector<int>>();
  cc->Inputs().Tag(kOutputEOSTag).Set<bool>();
  cc->Inputs().Tag(kOutputTokenIDsTag).Set<std::vector<int>>();

  return absl::OkStatus();
}

absl::Status GenerateTokensMetricsSinkCalculator::Open(mediapipe::CalculatorContext* cc) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  const auto& options = cc->Options<optionspb::GenerateTokensMetricsSinkCalculatorOptions>();

  metric_attributes_ = absl::flat_hash_map<std::string, std::string>(
      options.metric_attributes().begin(), options.metric_attributes().end());

  output_tokens_counter_ = metrics_system.GetOrCreateCounter(
      "gml_gem_pipe_gentokens_output",
      "Count of output tokens as they arrive. Better represents the arrival rate of output tokens"
      "than the histogram equivalent.");
  output_tokens_histogram_ = metrics_system.GetOrCreateHistogramWithBounds<uint64_t>(
      "gml_gem_pipe_gentokens_output_hist", "Histogram of output stream sizes.",
      metrics::kDefaultHistogramBounds);

  input_tokens_histogram_ = metrics_system.GetOrCreateHistogramWithBounds<uint64_t>(
      "gml_gem_pipe_gentokens_input_hist", "Histogram of input stream sizes.",
      metrics::kDefaultHistogramBounds);

  return absl::OkStatus();
}

void GenerateTokensMetricsSinkCalculator::HandleInput(mediapipe::CalculatorContext* cc) {
  auto input_tokens = cc->Inputs().Tag(kInputTokenIDsTag).Get<std::vector<int>>();
  input_tokens_histogram_->Record(input_tokens.size(), metric_attributes_);
}

void GenerateTokensMetricsSinkCalculator::HandleOutput(mediapipe::CalculatorContext* cc) {
  auto output_tokens_size = cc->Inputs().Tag(kOutputTokenIDsTag).Get<std::vector<int>>().size();
  auto output_eos_value = cc->Inputs().Tag(kOutputEOSTag).Get<bool>();
  output_tokens_counter_->Add(output_tokens_size, metric_attributes_);
  running_output_tokens_ += output_tokens_size;
  if (output_eos_value) {
    output_tokens_histogram_->Record(running_output_tokens_, metric_attributes_);
    running_output_tokens_ = 0;
  }
}

absl::Status GenerateTokensMetricsSinkCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (!cc->Inputs().Tag(kInputTokenIDsTag).IsEmpty()) {
    HandleInput(cc);
  }
  if (!cc->Inputs().Tag(kOutputTokenIDsTag).IsEmpty()) {
    HandleOutput(cc);
  }

  return absl::OkStatus();
}

REGISTER_CALCULATOR(GenerateTokensMetricsSinkCalculator);
}  // namespace gml::gem::calculators::core