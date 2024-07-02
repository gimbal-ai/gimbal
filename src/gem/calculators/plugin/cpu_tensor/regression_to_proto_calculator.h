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
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::RegressionToProtoOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
