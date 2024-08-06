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

#include "src/gem/calculators/plugin/cpu_tensor/tokens_to_tensor_calculator.h"

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;

constexpr std::string_view kTokensTag = "TOKENS";
constexpr std::string_view kTensorTag = "TENSOR";

absl::Status TokensToTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kTokensTag).Set<std::vector<int>>();
  cc->Outputs().Tag(kTensorTag).Set<CPUTensorPtr>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

Status TokensToTensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status TokensToTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                             ExecutionContext* exec_ctx) {
  const auto& tokens = cc->Inputs().Tag(kTokensTag).Get<std::vector<int>>();

  GML_ASSIGN_OR_RETURN(auto tensor, exec_ctx->TensorPool()->GetTensor(
                                        {1, static_cast<int>(tokens.size())}, DataType::INT64));

  std::memcpy(tensor->data(), tokens.data(), sizeof(int) * tokens.size());

  auto packet = mediapipe::MakePacket<CPUTensorPtr>(std::move(tensor));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kTensorTag).AddPacket(std::move(packet));
  return Status::OK();
}

Status TokensToTensorCalculator::CloseImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(TokensToTensorCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
