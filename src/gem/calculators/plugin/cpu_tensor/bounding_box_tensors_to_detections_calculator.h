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

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;

/**
 *  BoundingBoxTensorsToDetections Graph API:
 *
 *  Inputs:
 *    BOX_TENSOR (1 x N_CANDIDATES x 4) tensor containing candidate (a.k.a "anchor") boxes.
 *      Currently we only support (y1, x1, y2, x2) unnormalized bounding box coordinates. In the
 *      future, we'll likely support (xc, yc, w, h) format as well.
 *
 *    SCORE_TENSOR (1 x N_CLASSES x N_CANDIDATES) tensor containing scores per class per box.
 *
 *    INDEX_TENSOR (N_SELECTED x 3) tensor containing NMS selected indices of best candidates.
 *      Each index is (batch_idx, class_idx, box_idx).
 *
 *    ORIG_IMAGE_SHAPE (1 x 2) original image shape [width, height]
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
  Status CheckShapes(const TensorShape& boxes, const TensorShape& scores,
                     const TensorShape& indices, const TensorShape& original_shape);

 private:
  bool shapes_checked_ = false;

  optionspb::BoundingBoxToDetectionsOptions options_;

  using BoundingBoxConvFunc = std::function<void(
      const float*, const float*, ::gml::internal::api::core::v1::NormalizedCenterRect*)>;
  BoundingBoxConvFunc bounding_box_converter_;
};

}  // namespace gml::gem::calculators::cpu_tensor
