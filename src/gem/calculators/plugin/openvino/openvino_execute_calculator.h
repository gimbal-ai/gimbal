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
 *   RESET_STATE: reset the infer request state for stateful models.
 *  Outputs:
 *   Each output will be a CPUTensorPtr
 **/
class OpenVinoExecuteCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::OpenVinoExecuteCalculatorOptions options_;
  ov::InferRequest infer_request_;

  std::vector<int64_t> calc_input_idx_to_model_idx_;
  std::vector<int64_t> calc_output_idx_to_model_idx_;
  absl::flat_hash_map<int64_t, int64_t> loopback_mapping_;

  Status SetupIndices(mediapipe::CalculatorContext*);
};

}  // namespace gml::gem::calculators::openvino
