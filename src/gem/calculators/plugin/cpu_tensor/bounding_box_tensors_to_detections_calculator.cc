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

#include "src/gem/calculators/plugin/cpu_tensor/bounding_box_tensors_to_detections_calculator.h"

#include <mediapipe/framework/calculator_framework.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::calculators::cpu_tensor::optionspb::BoundingBoxToDetectionsOptions;
using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::NormalizedCenterRect;

constexpr std::string_view kBoxTag = "BOX_TENSOR";
constexpr std::string_view kScoreTag = "SCORE_TENSOR";
constexpr std::string_view kIndicesTag = "INDEX_TENSOR";
constexpr std::string_view kShapeTag = "ORIG_IMAGE_SHAPE";

namespace internal {
void ConvertDiagCornersYXToNormalizedCenter(const float* coords, const float* orig_image_shape,
                                            NormalizedCenterRect* rect) {
  auto y1 = coords[0];
  auto x1 = coords[1];
  auto y2 = coords[2];
  auto x2 = coords[3];

  auto orig_h = orig_image_shape[0];
  auto orig_w = orig_image_shape[1];

  auto xc = (x1 + x2) / 2;
  auto yc = (y1 + y2) / 2;
  auto w = x2 - x1;
  auto h = y2 - y1;

  rect->set_xc(xc / orig_w);
  rect->set_yc(yc / orig_h);
  rect->set_width(w / orig_w);
  rect->set_height(h / orig_h);
}

void ConvertCenterAnchoredToNormalizedCetner(const float* coords, const float* orig_image_shape,
                                             NormalizedCenterRect* rect) {
  auto xc = coords[0];
  auto yc = coords[1];
  auto w = coords[2];
  auto h = coords[3];

  auto orig_h = orig_image_shape[0];
  auto orig_w = orig_image_shape[1];

  rect->set_xc(xc / orig_w);
  rect->set_yc(yc / orig_h);
  rect->set_width(w / orig_w);
  rect->set_height(h / orig_h);
}

void ConvertNormalizedDiagCornerXYToNormalizedCenter(const float* coords, const float*,
                                                     NormalizedCenterRect* rect) {
  auto x1 = coords[0];
  auto y1 = coords[1];
  auto x2 = coords[2];
  auto y2 = coords[3];

  rect->set_xc((x1 + x2) / 2);
  rect->set_yc((y1 + y2) / 2);
  rect->set_width(x2 - x1);
  rect->set_height(y2 - y1);
}

}  // namespace internal

absl::Status BoundingBoxTensorsToDetections::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kBoxTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kScoreTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kIndicesTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kShapeTag).Set<CPUTensorPtr>();
  cc->Outputs().Index(0).Set<std::vector<Detection>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status BoundingBoxTensorsToDetections::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<BoundingBoxToDetectionsOptions>();
  switch (options_.coordinate_format()) {
    case BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_DIAG_CORNERS_YX:
      bounding_box_converter_ = &internal::ConvertDiagCornersYXToNormalizedCenter;
      break;
    case BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_CENTER_ANCHORED:
      bounding_box_converter_ = &internal::ConvertCenterAnchoredToNormalizedCetner;
      break;
    case BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_NORMALIZED_DIAG_CORNERS_XY:
      bounding_box_converter_ = &internal::ConvertNormalizedDiagCornerXYToNormalizedCenter;
      break;
    default:
      return AbslStatusAdapter(error::InvalidArgument("Unknown bounding box CoordinateFormat"));
  }
  return absl::OkStatus();
}

absl::Status BoundingBoxTensorsToDetections::Process(mediapipe::CalculatorContext* cc) {
  auto boxes = cc->Inputs().Tag(kBoxTag).Get<CPUTensorPtr>();
  auto scores = cc->Inputs().Tag(kScoreTag).Get<CPUTensorPtr>();
  auto indices = cc->Inputs().Tag(kIndicesTag).Get<CPUTensorPtr>();
  auto original_shape = cc->Inputs().Tag(kShapeTag).Get<CPUTensorPtr>();

  if (!shapes_checked_) {
    GML_ABSL_RETURN_IF_ERROR(
        CheckShapes(boxes->Shape(), scores->Shape(), indices->Shape(), original_shape->Shape()));
    // TODO(james): check data types.
    shapes_checked_ = true;
  }

  auto batches = boxes->Shape()[0];
  if (scores->Shape()[0] != batches || original_shape->Shape()[0] != batches) {
    return AbslStatusAdapter(error::InvalidArgument(
        "All inputs with a batch dimension must have the same batch dimension"));
  }
  if (batches != 1) {
    return AbslStatusAdapter(error::Unimplemented("Currently only batches=1 is supported"));
  }

  const auto* indices_data = indices->TypedData<DataType::INT32>();
  const auto index_step = indices->Shape()[1];

  const auto* box_data = boxes->TypedData<DataType::FLOAT32>();
  const auto box_step = boxes->Shape()[2];
  const auto box_batch_step = boxes->Shape()[1] * box_step;

  const auto* score_data = scores->TypedData<DataType::FLOAT32>();
  const auto score_step = scores->Shape()[2];
  const auto score_batch_step = scores->Shape()[1] * score_step;

  const auto* orig_shape_data = original_shape->TypedData<DataType::FLOAT32>();

  std::vector<Detection> detections;

  for (int index_idx = 0; index_idx < indices->Shape()[0]; ++index_idx) {
    const auto* index_triple = &indices_data[static_cast<ptrdiff_t>(index_idx * index_step)];
    const auto batch_idx = index_triple[0];
    const auto class_idx = index_triple[1];
    const auto box_idx = index_triple[2];

    if (batch_idx < 0 || class_idx < 0 || box_idx < 0) {
      // Openvino's GPU implementation of NonMaxSuppression outputs 1600 "selected" indices where
      // almost all of them are -1.
      continue;
    }
    auto& d = detections.emplace_back();

    const auto score = score_data[score_batch_step * batch_idx + score_step * class_idx + box_idx];
    auto* label = d.add_label();
    label->set_score(score);
    label->set_label(options_.index_to_label(class_idx));

    const auto* box_coords = &box_data[box_batch_step * batch_idx + box_step * box_idx];
    bounding_box_converter_(box_coords, orig_shape_data, d.mutable_bounding_box());
  }

  auto packet = mediapipe::MakePacket<std::vector<Detection>>(std::move(detections));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  return absl::OkStatus();
}

absl::Status BoundingBoxTensorsToDetections::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

Status BoundingBoxTensorsToDetections::CheckShapes(const TensorShape& boxes,
                                                   const TensorShape& scores,
                                                   const TensorShape& indices,
                                                   const TensorShape& original_shape) {
  // TODO(james): move to a unified shape checking system. Potentially as part of mediapipe's
  // GetContract.
  if (boxes.size() != 3) {
    return error::InvalidArgument("$0 must be 3-dimensional (received shape $1)", kBoxTag, boxes);
  }
  if (scores.size() != 3) {
    return error::InvalidArgument("$0 must be 3-dimensional (received shape $1)", kScoreTag,
                                  scores);
  }
  if (indices.size() != 2) {
    return error::InvalidArgument("$0 must be 2-dimensional (received shape $1)", kIndicesTag,
                                  indices);
  }
  if (original_shape.size() != 2) {
    return error::InvalidArgument("$0 must be 2-dimensional (received shape $1)", kShapeTag,
                                  original_shape);
  }

  if (boxes[2] != 4) {
    return error::InvalidArgument("$0's last dimension must be of size 4 (received shape $1",
                                  kBoxTag, boxes);
  }
  if (scores[1] != options_.index_to_label_size()) {
    return error::InvalidArgument(
        "$0 has $1 classes in dimension 1 but the options given to the calculator have $2 "
        "classes",
        kScoreTag, scores[1], options_.index_to_label_size());
  }

  if (boxes[1] != scores[2]) {
    return error::InvalidArgument("$0 and $1 disagree on N_CANDIDATES ($2 vs $3)", kBoxTag,
                                  kScoreTag, boxes[1], scores[2]);
  }

  if (indices[1] != 3) {
    return error::InvalidArgument("$0's last dimension must be of size 3 (received shape $1)",
                                  kIndicesTag, indices);
  }

  return Status::OK();
}

REGISTER_CALCULATOR(BoundingBoxTensorsToDetections);

}  // namespace gml::gem::calculators::cpu_tensor
