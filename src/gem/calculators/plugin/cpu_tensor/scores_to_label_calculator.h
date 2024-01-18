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

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/scores_to_label_calculator_options.pb.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 * ScoresToLabelCalculator Graph API:
 *
 * TODO(james): Currently this API modifies a Detection proto by changing a broad label into a more
 * specific one. We should probably reconsider our APIs and have this output a separate
 * "Classification" proto, that we can then merge with a Detection proto in a separate step.
 *
 *  Options:
 *    INDEX_TO_LABEL mapping from index to label.
 *  Inputs:
 *    SCORES CPUTensorPtr of shape NxC where N is batch size and C is number of classes.
 *    DETECTION Detection proto to add label to.
 *  Outputs:
 *    DETECTION Detection proto with label added.
 *
 */
class ScoresToLabelCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::ScoresToLabelCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
