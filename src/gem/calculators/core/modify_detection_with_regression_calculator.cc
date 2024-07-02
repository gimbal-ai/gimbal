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

absl::Status ModifyDetectionWithRegressionCalculator::Open(mediapipe::CalculatorContext*) {
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

absl::Status ModifyDetectionWithRegressionCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ModifyDetectionWithRegressionCalculator);

}  // namespace gml::gem::calculators::core
