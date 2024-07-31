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

#pragma once

#include <mediapipe/framework/calculator_framework.h>

#include "src/api/corepb/v1/mediastream.pb.h"

namespace gml::gem::calculators::core {

/**
 *  ModifyDetectionWithClassificationCalculator Graph API:
 *  Sets the Detection label to the Classification label, but preserves the Detection confidence
 * score.
 *
 *  Inputs:
 *    DETECTION internal::api::corepb::v1::Detection proto.
 *    CLASSIFICATION internal::api::corepb::v1::Classification proto.
 *
 *  Outputs:
 *    DETECTION modified internal::api::corepb::v1::Detection proto
 */
class ModifyDetectionWithClassificationCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;
};

}  // namespace gml::gem::calculators::core
