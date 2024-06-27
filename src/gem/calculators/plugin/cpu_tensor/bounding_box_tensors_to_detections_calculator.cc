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

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::calculators::cpu_tensor::optionspb::BoundingBoxToDetectionsOptions;
using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kBoxTag = "BOX_TENSOR";
constexpr std::string_view kScoreTag = "SCORE_TENSOR";
constexpr std::string_view kClassTag = "CLASS_TENSOR";

absl::Status BoundingBoxTensorsToDetections::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kBoxTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kScoreTag).Set<CPUTensorPtr>();
  cc->Inputs().Tag(kClassTag).Set<CPUTensorPtr>();
  cc->Outputs().Index(0).Set<std::vector<Detection>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status BoundingBoxTensorsToDetections::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<BoundingBoxToDetectionsOptions>();
  return absl::OkStatus();
}

absl::Status BoundingBoxTensorsToDetections::Process(mediapipe::CalculatorContext* cc) {
  auto boxes = cc->Inputs().Tag(kBoxTag).Get<CPUTensorPtr>();
  auto scores = cc->Inputs().Tag(kScoreTag).Get<CPUTensorPtr>();
  auto classes = cc->Inputs().Tag(kClassTag).Get<CPUTensorPtr>();

  if (!shapes_checked_) {
    GML_ABSL_RETURN_IF_ERROR(CheckShapes(boxes->Shape(), scores->Shape(), classes->Shape()));
    // TODO(james): check data types.
    shapes_checked_ = true;
  }

  auto batches = boxes->Shape()[0];
  if (scores->Shape()[0] != batches || classes->Shape()[0] != batches) {
    return AbslStatusAdapter(error::InvalidArgument(
        "All inputs with a batch dimension must have the same batch dimension"));
  }
  if (batches != 1) {
    return AbslStatusAdapter(error::Unimplemented("Currently only batches=1 is supported"));
  }

  size_t num_selected = boxes->Shape()[1];

  const auto* box_data = boxes->TypedData<DataType::FLOAT32>();
  const auto box_step = boxes->Shape()[2];

  const auto* score_data = scores->TypedData<DataType::FLOAT32>();
  std::function<int(size_t)> get_class_idx;
  if (classes->DataType() == DataType::INT32) {
    get_class_idx = [&](size_t idx) {
      return static_cast<int>(classes->TypedData<DataType::INT32>()[idx]);
    };
  } else {
    get_class_idx = [&](size_t idx) {
      return static_cast<int>(classes->TypedData<DataType::INT64>()[idx]);
    };
  }

  std::vector<Detection> detections;

  for (size_t i = 0; i < num_selected; ++i) {
    const auto score = score_data[i];
    const auto class_idx = get_class_idx(i);

    if (class_idx < 0) {
      continue;
    }

    auto class_label = options_.index_to_label(class_idx);
    if (class_label == "") {
      continue;
    }
    auto& d = detections.emplace_back();
    auto* label = d.add_label();
    label->set_score(score);
    label->set_label(class_label);

    auto* bbox = d.mutable_bounding_box();
    bbox->set_xc(box_data[box_step * i + 0]);
    bbox->set_yc(box_data[box_step * i + 1]);
    bbox->set_width(box_data[box_step * i + 2]);
    bbox->set_height(box_data[box_step * i + 3]);
  }

  if (detections.empty()) {
    cc->Outputs().Index(0).SetNextTimestampBound(cc->InputTimestamp().NextAllowedInStream());
    return absl::OkStatus();
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
                                                   const TensorShape& classes) {
  // TODO(james): move to a unified shape checking system. Potentially as part of mediapipe's
  // GetContract.
  if (boxes.size() != 3) {
    return error::InvalidArgument("$0 must be 3-dimensional (received shape $1)", kBoxTag, boxes);
  }
  if (scores.size() != 2) {
    return error::InvalidArgument("$0 must be 2-dimensional (received shape $1)", kScoreTag,
                                  scores);
  }
  if (classes.size() != 2) {
    return error::InvalidArgument("$0 must be 2-dimensional (received shape $1)", kClassTag,
                                  classes);
  }

  if (boxes[2] != 4) {
    return error::InvalidArgument("$0's last dimension must be of size 4 (received shape $1",
                                  kBoxTag, boxes);
  }

  if (boxes[1] != scores[1]) {
    return error::InvalidArgument("$0 and $1 disagree on N_SELECTED ($2 vs $3)", kBoxTag, kScoreTag,
                                  boxes[1], scores[1]);
  }

  if (boxes[1] != classes[1]) {
    return error::InvalidArgument("$0 and $1 disagree on N_SELECTED ($2 vs $3)", kBoxTag, kClassTag,
                                  boxes[1], classes[1]);
  }

  return Status::OK();
}

REGISTER_CALCULATOR(BoundingBoxTensorsToDetections);

}  // namespace gml::gem::calculators::cpu_tensor
