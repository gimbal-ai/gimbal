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

  auto num_classes = mask_tensor->Shape()[1];
  if (num_classes != options_.index_to_label_size()) {
    return AbslStatusAdapter(error::InvalidArgument(
        "number of classes in options does not match mask tensor channel dimension"));
  }

  auto height = mask_tensor->Shape()[2];
  auto width = mask_tensor->Shape()[3];

  const auto* mask_data = mask_tensor->TypedData<DataType::INT8>();
  const auto class_step = height * width;

  Segmentation segmentation;
  segmentation.set_height(height);
  segmentation.set_width(width);

  for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
    const auto* class_data = mask_data + (class_step * static_cast<size_t>(class_idx));
    auto class_label = options_.index_to_label(class_idx);
    auto* mask = segmentation.add_masks();
    mask->set_label(class_label);
    int32_t count = 0;
    bool current = false;
    for (int j = 0; j < class_step; ++j) {
      if (static_cast<bool>(class_data[j]) ^ current) {
        mask->add_run_length_encoding(count);
        count = 0;
        current = class_data[j];
      }
      count++;
    }
    mask->add_run_length_encoding(count);
    if (!current) {
      // Make sure size of run length encoding is always even.
      mask->add_run_length_encoding(0);
    }
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
