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

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/calculators/plugin/cpu_tensor/base.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/regression_to_proto_options.pb.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 *  RegressionToProto Graph API:
 *
 *  Inputs:
 *    TENSOR (B) float tensor.
 *
 *  Outputs:
 *    core::v1::Regression a regression proto with the regression value.
 */
class RegressionToProtoCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::RegressionToProtoOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
