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

#include "src/gem/calculators/core/buffer_tokens_for_autoregression_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kTokensTag = "TOKENS";
constexpr std::string_view kLoopStartTag = "LOOP_START";
constexpr std::string_view kAllTokensTag = "ALL_TOKENS";

absl::Status BufferTokensForAutoregressionCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kTokensTag).Set<std::vector<int>>();
  cc->Inputs().Tag(kLoopStartTag).Set<bool>();
  cc->Outputs().Tag(kAllTokensTag).Set<std::vector<int>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status BufferTokensForAutoregressionCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& tokens = cc->Inputs().Tag(kTokensTag).Get<std::vector<int>>();
  auto loop_start = cc->Inputs().Tag(kLoopStartTag).Get<bool>();

  if (loop_start) {
    all_tokens_.clear();
  }
  for (auto tok : tokens) {
    all_tokens_.push_back(tok);
  }

  cc->Outputs()
      .Tag(kAllTokensTag)
      .AddPacket(mediapipe::MakePacket<std::vector<int>>(all_tokens_).At(cc->InputTimestamp()));
  return absl::OkStatus();
}

REGISTER_CALCULATOR(BufferTokensForAutoregressionCalculator);

}  // namespace gml::gem::calculators::core
