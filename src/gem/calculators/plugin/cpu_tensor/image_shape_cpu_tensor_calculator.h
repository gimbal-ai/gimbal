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

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/cpu_tensor/base.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 * ImageShapeCPUTensorCalculator Graph API:
 *
 *  Input Side Packets:
 *    EXEC_CTX cputensor::ExecutionContext*
 *  Inputs:
 *    IMAGE_FRAME mediapipe::ImageFrame
 *  Outputs:
 *    IMAGE_SHAPE CPUTensorPtr containing the input image width and height ([h, w]).
 *
 */
class ImageShapeCPUTensorCalculator : public ExecutionContextBaseCalculator {
 public:
  using ExecutionContext = ::gml::gem::exec::cpu_tensor::ExecutionContext;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;
  Status ProcessImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;
  Status CloseImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;
};

}  // namespace gml::gem::calculators::cpu_tensor
