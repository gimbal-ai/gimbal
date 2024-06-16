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
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 protected:
  Status CheckShapes(const exec::core::TensorShape& boxes, const exec::core::TensorShape& scores,
                     const exec::core::TensorShape& classes);

 private:
  bool shapes_checked_ = false;
  optionspb::BoundingBoxToDetectionsOptions options_;
};

}  // namespace gml::gem::calculators::cpu_tensor
