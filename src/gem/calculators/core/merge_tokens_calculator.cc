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

#include "src/gem/calculators/core/merge_tokens_calculator.h"

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/stream_handler/sync_set_input_stream_handler.h>

namespace gml::gem::calculators::core {

constexpr std::string_view kInputStreamTag = "INPUT_TOKENS";
constexpr std::string_view kOutputStreamTag = "OUTPUT_TOKENS";
constexpr std::string_view kOutputEOSTag = "OUTPUT_EOS";

constexpr std::string_view kLoopStartTag = "LOOP_START";
constexpr std::string_view kMergedStreamTag = "MERGED_TOKENS";

absl::Status MergeTokensCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kInputStreamTag).Set<std::vector<int64_t>>();
  cc->Inputs().Tag(kOutputStreamTag).Set<std::vector<int64_t>>();
  cc->Inputs().Tag(kOutputEOSTag).Set<bool>();

  cc->Outputs().Tag(kMergedStreamTag).Set<std::vector<int64_t>>();
  cc->Outputs().Tag(kLoopStartTag).Set<bool>();

  return absl::OkStatus();
}

absl::Status MergeTokensCalculator::Open(mediapipe::CalculatorContext*) { return absl::OkStatus(); }

void MergeTokensCalculator::EmitInput(mediapipe::CalculatorContext* cc) {
  ++internal_timestamp_;
  cc->Outputs()
      .Tag(kMergedStreamTag)
      .AddPacket(mediapipe::MakePacket<std::vector<int64_t>>(input_buffer_.front())
                     .At(internal_timestamp_));
  cc->Outputs()
      .Tag(kLoopStartTag)
      .AddPacket(mediapipe::MakePacket<bool>(true).At(internal_timestamp_));
  input_buffer_.pop_front();
  is_processing_output_ = true;
}

void MergeTokensCalculator::HandleInput(mediapipe::CalculatorContext* cc) {
  const auto& input_stream_value = cc->Inputs().Tag(kInputStreamTag).Get<std::vector<int64_t>>();
  input_buffer_.push_back(input_stream_value);

  if (is_processing_output_) {
    return;
  }

  EmitInput(cc);
}

void MergeTokensCalculator::HandleOutput(mediapipe::CalculatorContext* cc) {
  const auto& output_eos_value = cc->Inputs().Tag(kOutputEOSTag).Get<bool>();

  if (output_eos_value) {
    is_processing_output_ = false;
    if (!input_buffer_.empty()) {
      EmitInput(cc);
    }
    return;
  }

  const auto& output_stream_value = cc->Inputs().Tag(kOutputStreamTag).Get<std::vector<int64_t>>();

  ++internal_timestamp_;
  cc->Outputs()
      .Tag(kMergedStreamTag)
      .AddPacket(
          mediapipe::MakePacket<std::vector<int64_t>>(output_stream_value).At(internal_timestamp_));
  cc->Outputs()
      .Tag(kLoopStartTag)
      .AddPacket(mediapipe::MakePacket<bool>(false).At(internal_timestamp_));
}

absl::Status MergeTokensCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (!cc->Inputs().Tag(kInputStreamTag).IsEmpty()) {
    HandleInput(cc);
  }

  if (!cc->Inputs().Tag(kOutputStreamTag).IsEmpty()) {
    HandleOutput(cc);
  }

  return absl::OkStatus();
}

absl::Status MergeTokensCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(MergeTokensCalculator);

}  // namespace gml::gem::calculators::core
