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

#include "src/gem/calculators/plugin/cpu_tensor/tensor_to_tokens_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;

constexpr std::string_view kTensorTag = "TENSOR";
constexpr std::string_view kTokensTag = "TOKENS";

absl::Status TensorToTokensCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorTag).Set<CPUTensorPtr>();
  cc->Outputs().Tag(kTokensTag).Set<std::vector<int>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status TensorToTokensCalculator::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status TensorToTokensCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& tensor = cc->Inputs().Tag(kTensorTag).Get<CPUTensorPtr>();

  if (tensor->DataType() != DataType::INT32) {
    return {absl::StatusCode::kInvalidArgument,
            absl::Substitute("expected tokens tensor to be INT32, received: $0",
                             magic_enum::enum_name(tensor->DataType()))};
  }

  auto tokens = std::make_unique<std::vector<int>>(tensor->size() / sizeof(int));
  std::memcpy(tokens->data(), tensor->data(), tensor->size());

  cc->Outputs().Tag(kTokensTag).Add(tokens.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status TensorToTokensCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(TensorToTokensCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
