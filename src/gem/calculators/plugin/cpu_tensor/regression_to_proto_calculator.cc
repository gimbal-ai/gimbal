/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
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

absl::Status RegressionToProtoCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(RegressionToProtoCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
