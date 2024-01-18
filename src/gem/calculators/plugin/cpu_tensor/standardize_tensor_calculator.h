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
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/standardize_tensor_calculator_options.pb.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 * StandardizeTensorCalculator Graph API:
 *
 *  Options:
 *    MEAN list of per-channel means to use for standardization.
 *    STDDEV list of per-channel stddev to use for standardization.
 *  Inputs:
 *    TENSOR CPUTensorPtr of shape NxCxHxW.
 *  Outputs:
 *    TENSOR CPUTensorPtr standardized version of the input tensor. Currently,
 *      this updates the tensor inplace, which is a little unsafe.
 *
 */
class StandardizeTensorCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::StandardizeTensorCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
