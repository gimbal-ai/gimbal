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

#include "src/gem/calculators/plugin/cpu_tensor/regression_to_proto_calculator.h"

#include <mediapipe/framework/calculator_framework.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/error.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::calculators::cpu_tensor::optionspb::RegressionToProtoOptions;
using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Regression;

constexpr std::string_view kTensorTag = "TENSOR";
constexpr std::string_view kRegressionTag = "REGRESSION";

absl::Status RegressionToProtoCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorTag).Set<CPUTensorPtr>();
  cc->Outputs().Tag(kRegressionTag).Set<Regression>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status RegressionToProtoCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<RegressionToProtoOptions>();
  return absl::OkStatus();
}

absl::Status RegressionToProtoCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto mask_tensor = cc->Inputs().Tag(kTensorTag).Get<CPUTensorPtr>();

  auto batches = mask_tensor->Shape()[0];
  if (batches != 1) {
    return AbslStatusAdapter(error::Unimplemented("Currently only batches=1 is supported"));
  }
  const auto* data = mask_tensor->TypedData<DataType::FLOAT32>();
  auto val = data[0];

  Regression regression;
  regression.set_label(options_.label());
  regression.set_value(val);

  auto packet = mediapipe::MakePacket<Regression>(std::move(regression));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kRegressionTag).AddPacket(std::move(packet));

  return absl::OkStatus();
}

REGISTER_CALCULATOR(RegressionToProtoCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
