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

#include "src/gem/calculators/plugin/cpu_tensor/scores_to_classification_calculator.h"

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/scores_to_classification_calculator_options.pb.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Classification;

constexpr std::string_view kScoresTag = "SCORES";
constexpr std::string_view kClassificationTag = "CLASSIFICATION";

namespace gml::gem::calculators::cpu_tensor {

absl::Status ScoresToClassificationCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kScoresTag).Set<CPUTensorPtr>();
  cc->Outputs().Tag(kClassificationTag).Set<Classification>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ScoresToClassificationCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<optionspb::ScoresToClassificationCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status ScoresToClassificationCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto scores = cc->Inputs().Tag(kScoresTag).Get<CPUTensorPtr>();

  if (scores->Shape().size() != 2) {
    return AbslStatusAdapter(error::InvalidArgument(
        "incorrect shape passed to ScoresToClassificationCalculator, expected NC, received [$0]",
        absl::StrJoin(scores->Shape(), ",")));
  }

  if (scores->Shape()[1] != options_.index_to_label_size()) {
    return AbslStatusAdapter(error::InvalidArgument(
        "second dimension of the scores tensor must be the same size as "
        "index_to_label in ScoresToClassificationCalculatorOptions, $0 vs $1",
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

  Classification classification;
  auto label = classification.add_label();
  label->set_label(options_.index_to_label(argmax));
  label->set_score(max_score);

  cc->Outputs()
      .Tag(kClassificationTag)
      .AddPacket(mediapipe::MakePacket<Classification>(classification).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

REGISTER_CALCULATOR(ScoresToClassificationCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
