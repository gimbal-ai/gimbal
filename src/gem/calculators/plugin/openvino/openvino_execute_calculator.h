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

#include <mediapipe/framework/calculator_framework.h>
#include "src/gem/calculators/plugin/openvino/optionspb/openvino_execute_calculator_options.pb.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"
#include "src/gem/exec/plugin/openvino/context.h"

namespace gml::gem::calculators::openvino {

/**
 * OpenVinoExecuteCalculator Graph API:
 *  Input Side Packets:
 *   ExecutionContext tagged with EXEC_CTX
 *  Inputs:
 *   Each input must be a CPUTensorPtr
 *  Outputs:
 *   Each output will be a CPUTensorPtr
 **/
class OpenVinoExecuteCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::OpenVinoExecuteCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::openvino
