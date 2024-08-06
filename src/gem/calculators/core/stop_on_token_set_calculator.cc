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

#include "src/gem/calculators/core/stop_on_token_set_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/calculators/core/optionspb/stop_on_token_set_calculator_options.pb.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kTokensTag = "TOKENS";
constexpr std::string_view kEOSTag = "EOS";

absl::Status StopOnTokenSetCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kTokensTag).Set<std::vector<int>>();
  cc->Outputs().Tag(kEOSTag).Set<bool>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status StopOnTokenSetCalculator::Open(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::StopOnTokenSetCalculatorOptions>();
  max_tokens_before_eos_ = options.max_tokens_before_eos();
  tokens_since_eos_ = 0;
  eos_tokens_.insert(options.eos_tokens().begin(), options.eos_tokens().end());
  return absl::OkStatus();
}

absl::Status StopOnTokenSetCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& tokens = cc->Inputs().Tag(kTokensTag).Get<std::vector<int>>();

  auto eos = std::make_unique<bool>();
  for (const auto& tok : tokens) {
    if (eos_tokens_.contains(tok)) {
      *eos = true;
      break;
    }
    tokens_since_eos_++;
    if (tokens_since_eos_ >= max_tokens_before_eos_) {
      *eos = true;
      break;
    }
  }

  if (*eos) {
    tokens_since_eos_ = 0;
  }

  cc->Outputs().Tag(kEOSTag).Add(eos.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status StopOnTokenSetCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(StopOnTokenSetCalculator);

}  // namespace gml::gem::calculators::core
