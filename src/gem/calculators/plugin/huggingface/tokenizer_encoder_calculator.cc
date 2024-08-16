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

#include "src/gem/calculators/plugin/huggingface/tokenizer_encoder_calculator.h"

#include <mediapipe/framework/calculator_registry.h>

#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/tensor.h"

namespace gml::gem::calculators::huggingface {

constexpr std::string_view kTokenizerContextTag = "EXEC_CTX";
constexpr std::string_view kTextTag = "TEXT";
constexpr std::string_view kTokenIDsTag = "TOKEN_IDS";

absl::Status TokenizerEncoderCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->InputSidePackets().Tag(kTokenizerContextTag).Set<exec::core::ExecutionContext*>();

  cc->Inputs().Tag(kTextTag).Set<std::string>();
  cc->Outputs().Tag(kTokenIDsTag).Set<std::vector<int>>();

  cc->SetTimestampOffset(0);

  return absl::OkStatus();
}

absl::Status TokenizerEncoderCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto* tokenizer_exec_ctx = static_cast<exec::huggingface::ExecutionContext*>(
      cc->InputSidePackets().Tag(kTokenizerContextTag).Get<exec::core::ExecutionContext*>());
  auto tokenizer = tokenizer_exec_ctx->Tokenizer();

  auto text = cc->Inputs().Tag(kTextTag).Get<std::string>();
  auto token_ids = tokenizer->Encode(text);
  cc->Outputs()
      .Tag(kTokenIDsTag)
      .AddPacket(
          mediapipe::MakePacket<std::vector<int>>(std::move(token_ids)).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

REGISTER_CALCULATOR(TokenizerEncoderCalculator);

}  // namespace gml::gem::calculators::huggingface
