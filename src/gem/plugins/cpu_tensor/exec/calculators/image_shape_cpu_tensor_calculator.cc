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

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "src/common/base/base.h"
#include "src/gem/plugins/cpu_tensor/exec/calculators/image_shape_cpu_tensor_calculator.h"
#include "src/gem/plugins/cpu_tensor/exec/context.h"

namespace gml {
namespace gem {
namespace cputensor {
namespace calculators {

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kOutputTag = "IMAGE_SHAPE";

absl::Status ImageShapeCPUTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kOutputTag).Set<CPUTensorPtr>();
  return absl::OkStatus();
}

Status ImageShapeCPUTensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status ImageShapeCPUTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                  ExecutionContext* exec_ctx) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  GML_ASSIGN_OR_RETURN(auto cpu_tensor, exec_ctx->TensorPool()->GetTensor(sizeof(float) * 2));

  float* shape = reinterpret_cast<float*>(cpu_tensor->data());

  shape[0] = static_cast<float>(image_frame.Height());
  shape[1] = static_cast<float>(image_frame.Width());

  auto packet = mediapipe::MakePacket<CPUTensorPtr>(std::move(cpu_tensor));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kOutputTag).AddPacket(std::move(packet));
  return Status::OK();
}

Status ImageShapeCPUTensorCalculator::CloseImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(ImageShapeCPUTensorCalculator);

}  // namespace calculators
}  // namespace cputensor
}  // namespace gem
}  // namespace gml
