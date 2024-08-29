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

#pragma once

#include <mediapipe/framework/calculator_framework.h>

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/scores_to_classification_calculator_options.pb.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 * ScoresToClassificationCalculator Graph API:
 *
 *
 *  Options:
 *    INDEX_TO_LABEL mapping from index to label.
 *  Inputs:
 *    SCORES CPUTensorPtr of shape NxC where N is batch size and C is number of classes.
 *  Outputs:
 *    CLASSIFICATION Classification proto containing the label.
 *
 */
class ScoresToClassificationCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::ScoresToClassificationCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
