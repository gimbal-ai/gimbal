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

#include "src/gem/calculators/plugin/cpu_tensor/scores_to_label_calculator.h"

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/scores_to_label_calculator_options.pb.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kScoresTag = "SCORES";
constexpr std::string_view kDetectionTag = "DETECTION";

namespace gml::gem::calculators::cpu_tensor {

absl::Status ScoresToLabelCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kScoresTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  cc->Outputs().Tag(kDetectionTag).Set<Detection>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ScoresToLabelCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<optionspb::ScoresToLabelCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status ScoresToLabelCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto scores = cc->Inputs().Tag(kScoresTag).Get<CPUTensorPtr>();

  if (scores->Shape().size() != 2) {
    return AbslStatusAdapter(error::InvalidArgument(
        "incorrect shape passed to ScoresToLabelCalculator, expected NC, received [$0]",
        absl::StrJoin(scores->Shape(), ",")));
  }

  if (scores->Shape()[1] != options_.index_to_label_size()) {
    return AbslStatusAdapter(
        error::InvalidArgument("second dimension of the scores tensor must be the same size as "
                               "index_to_label in ScoresToLabelCalculatorOptions, $0 vs $1",
                               scores->Shape()[1], options_.index_to_label_size()));
  }

  auto B = scores->Shape()[0];
  if (B != 1) {
    return AbslStatusAdapter(error::Unimplemented("batches != 1 not implemented"));
  }
  auto C = scores->Shape()[1];

  auto* data = scores->TypedData<DataType::FLOAT32>();
  int argmax = 0;
  float max_score = 0;
  for (int idx = 0; idx < C; ++idx) {
    if (data[idx] > max_score) {
      max_score = data[idx];
      argmax = idx;
    }
  }

  // Intentionally copying the detection here, to avoid issues with mutating inputs.
  Detection detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();
  detection.mutable_label(0)->set_label(options_.index_to_label(argmax));

  cc->Outputs()
      .Tag(kDetectionTag)
      .AddPacket(mediapipe::MakePacket<Detection>(detection).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

REGISTER_CALCULATOR(ScoresToLabelCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
