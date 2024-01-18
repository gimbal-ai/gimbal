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

#include "src/gem/calculators/plugin/cpu_tensor/standardize_tensor_calculator.h"

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/standardize_tensor_calculator_options.pb.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;

constexpr std::string_view kTensorTag = "TENSOR";

namespace gml::gem::calculators::cpu_tensor {

absl::Status StandardizeTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorTag).Set<CPUTensorPtr>();
  cc->Outputs().Tag(kTensorTag).Set<CPUTensorPtr>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status StandardizeTensorCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<optionspb::StandardizeTensorCalculatorOptions>();

  for (const auto& stddev : options_.stddev()) {
    if (stddev == 0) {
      return AbslStatusAdapter(error::InvalidArgument("all stddevs must be non-zero, received: []",
                                                      absl::StrJoin(options_.stddev(), ",")));
    }
  }
  return absl::OkStatus();
}

absl::Status StandardizeTensorCalculator::Process(mediapipe::CalculatorContext* cc) {
  CPUTensorPtr tensor = cc->Inputs().Tag(kTensorTag).Get<CPUTensorPtr>();

  if (tensor->Shape().size() != 4) {
    return AbslStatusAdapter(error::InvalidArgument(
        "incorrect shape passed to StandardizeTensorCalculator, expected NCHW, received [$0]",
        absl::StrJoin(tensor->Shape(), ",")));
  }

  auto B = tensor->Shape()[0];
  auto C = tensor->Shape()[1];
  auto H = tensor->Shape()[2];
  auto W = tensor->Shape()[3];

  if (options_.mean_size() != C || options_.stddev_size() != C) {
    return AbslStatusAdapter(
        error::InvalidArgument("per-channel means/stddevs passed to StandardizeTensorCalculator "
                               "must match the tensor's number of channels, ($0/$1) vs ($2)",
                               options_.mean_size(), options_.stddev_size(), C));
  }

  // TODO(james): this is not safe. If multiple calculators use the input tensor, they will end up
  // potentially using the modifications here. For now, the graph is structured such that the input
  // to this calculator is never reused, but if that changes there will be issues.
  auto* data = tensor->TypedData<DataType::FLOAT32>();
  for (int chan_idx = 0; chan_idx < C; ++chan_idx) {
    auto mean = options_.mean(chan_idx);
    auto stddev = options_.stddev(chan_idx);
    for (int batch_idx = 0; batch_idx < B; ++batch_idx) {
      for (int row_idx = 0; row_idx < H; ++row_idx) {
        for (int col_idx = 0; col_idx < W; ++col_idx) {
          auto idx = batch_idx * C * H * W + chan_idx * H * W + row_idx * W + col_idx;
          data[idx] = (data[idx] - mean) / stddev;
        }
      }
    }
  }
  cc->Outputs()
      .Tag(kTensorTag)
      .AddPacket(mediapipe::MakePacket<CPUTensorPtr>(std::move(tensor)).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status StandardizeTensorCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(StandardizeTensorCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
