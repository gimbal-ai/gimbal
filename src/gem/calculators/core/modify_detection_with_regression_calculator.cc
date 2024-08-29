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

#include "src/gem/calculators/core/modify_detection_with_regression_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/detection.pb.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::Regression;

constexpr std::string_view kDetectionTag = "DETECTION";
constexpr std::string_view kRegressionTag = "REGRESSION";

absl::Status ModifyDetectionWithRegressionCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  cc->Inputs().Tag(kRegressionTag).Set<Regression>();
  cc->Outputs().Tag(kDetectionTag).Set<Detection>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ModifyDetectionWithRegressionCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();
  const auto& regression = cc->Inputs().Tag(kRegressionTag).Get<Regression>();
  auto modified_detection = std::make_unique<Detection>(detection);
  auto* regression_label = modified_detection->add_label();
  regression_label->set_label(absl::StrFormat("%s: %.2f", regression.label(), regression.value()));
  cc->Outputs().Tag(kDetectionTag).Add(modified_detection.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ModifyDetectionWithRegressionCalculator);

}  // namespace gml::gem::calculators::core
