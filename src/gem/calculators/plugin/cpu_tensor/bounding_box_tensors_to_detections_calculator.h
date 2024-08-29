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

#pragma once

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/calculators/plugin/cpu_tensor/base.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/bounding_box_tensors_to_detections_options.pb.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

/**
 *  BoundingBoxTensorsToDetections Graph API:
 *
 *  Inputs:
 *    BOX_TENSOR (1 x N_SELECTED x 4) float32 tensor containing selected  boxes, in (xc, yc, w, h)
 * normalized format as well.
 *
 *    SCORE_TENSOR (1 x N_SELECTED) int tensor containing confidence scores for each prediction.
 *
 *    CLASS_TENSOR (1 x N_SELECTED) int tensor containing class indices for each prediction.
 *
 *  Outputs:
 *    std::vector<Detection> a list of detection protos for the image.
 */
class BoundingBoxTensorsToDetections : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 protected:
  Status CheckShapes(const exec::core::TensorShape& boxes, const exec::core::TensorShape& scores,
                     const exec::core::TensorShape& classes);

 private:
  bool shapes_checked_ = false;
  optionspb::BoundingBoxToDetectionsOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
