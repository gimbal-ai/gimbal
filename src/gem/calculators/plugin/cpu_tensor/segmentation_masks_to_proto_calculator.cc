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

#include "src/gem/calculators/plugin/cpu_tensor/segmentation_masks_to_proto_calculator.h"

#include <mediapipe/framework/calculator_framework.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/error.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::calculators::cpu_tensor::optionspb::SegmentationMasksToProtoOptions;
using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::internal::api::core::v1::Segmentation;

constexpr std::string_view kMaskTag = "MASK_TENSOR";

namespace {

struct RLEState {
  int32_t last_index = -1;
  std::vector<int32_t> rle;
};

template <typename T>
void MaskTensorToProto(const T* data, int32_t height, int32_t width,
                       const SegmentationMasksToProtoOptions& options, Segmentation* proto) {
  std::vector<RLEState> state_per_class(options.index_to_label_size());

  int32_t last_class_idx = -1;

  for (int i = 0; i < height * width; ++i) {
    auto class_idx = data[i];
    if (class_idx < 0) {
      continue;
    }
    auto& state = state_per_class[class_idx];
    if (class_idx == last_class_idx) {
      state.rle.back()++;
      state.last_index = i;
      continue;
    }
    auto zeros = (i - 1) - state.last_index;
    state.rle.push_back(zeros);
    state.rle.push_back(1);
    state.last_index = i;
    last_class_idx = class_idx;
  }

  for (auto& state : state_per_class) {
    int32_t zeros = (height * width - 1) - state.last_index;
    if (zeros <= 0) {
      continue;
    }
    state.rle.push_back(zeros);
    state.rle.push_back(0);
  }

  for (const auto& [class_idx, state] : Enumerate(state_per_class)) {
    auto* mask = proto->add_masks();
    auto label = options.index_to_label(static_cast<int32_t>(class_idx));
    mask->set_label(label);
    mask->mutable_run_length_encoding()->Assign(state.rle.begin(), state.rle.end());
  }
}
}  // namespace

absl::Status SegmentationMasksToProtoCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kMaskTag).Set<CPUTensorPtr>();
  cc->Outputs().Index(0).Set<Segmentation>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status SegmentationMasksToProtoCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<SegmentationMasksToProtoOptions>();
  return absl::OkStatus();
}

absl::Status SegmentationMasksToProtoCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto mask_tensor = cc->Inputs().Tag(kMaskTag).Get<CPUTensorPtr>();

  auto batches = mask_tensor->Shape()[0];
  if (batches != 1) {
    return AbslStatusAdapter(error::Unimplemented("Currently only batches=1 is supported"));
  }

  if (mask_tensor->Shape()[1] != 1) {
    return AbslStatusAdapter(error::InvalidArgument(
        "Expected Segmentation masks in [B, 1, H, W] format, received size $0 second dimension",
        mask_tensor->Shape()[1]));
  }

  auto height = mask_tensor->Shape()[2];
  auto width = mask_tensor->Shape()[3];

  Segmentation segmentation;
  segmentation.set_height(height);
  segmentation.set_width(width);

  if (mask_tensor->DataType() == DataType::INT32) {
    const auto* data = mask_tensor->TypedData<DataType::INT32>();
    MaskTensorToProto(data, height, width, options_, &segmentation);
  } else {
    const auto* data = mask_tensor->TypedData<DataType::INT64>();
    MaskTensorToProto(data, height, width, options_, &segmentation);
  }

  auto packet = mediapipe::MakePacket<Segmentation>(std::move(segmentation));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  return absl::OkStatus();
}

absl::Status SegmentationMasksToProtoCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(SegmentationMasksToProtoCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
