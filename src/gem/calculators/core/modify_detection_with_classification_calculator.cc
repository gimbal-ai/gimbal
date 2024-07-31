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

#include "src/gem/calculators/core/modify_detection_with_classification_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/detection.pb.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Classification;
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kDetectionTag = "DETECTION";
constexpr std::string_view kClassificationTag = "CLASSIFICATION";

absl::Status ModifyDetectionWithClassificationCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  cc->Inputs().Tag(kClassificationTag).Set<Classification>();
  cc->Outputs().Tag(kDetectionTag).Set<Detection>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ModifyDetectionWithClassificationCalculator::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status ModifyDetectionWithClassificationCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  const auto& detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();
  const auto& classification = cc->Inputs().Tag(kClassificationTag).Get<Classification>();
  auto modified_detection = std::make_unique<Detection>(detection);
  float score = modified_detection->label()[0].score();
  modified_detection->clear_label();
  auto label = modified_detection->add_label();
  label->set_label(classification.label()[0].label());
  label->set_score(score);

  cc->Outputs().Tag(kDetectionTag).Add(modified_detection.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status ModifyDetectionWithClassificationCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ModifyDetectionWithClassificationCalculator);

}  // namespace gml::gem::calculators::core
