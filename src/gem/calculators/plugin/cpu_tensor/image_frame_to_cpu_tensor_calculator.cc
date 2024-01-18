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

#include "src/gem/calculators/plugin/cpu_tensor/image_frame_to_cpu_tensor_calculator.h"

#include <cstddef>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "src/common/base/base.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kOutputTag = "CPU_TENSOR";

absl::Status ImageFrameToCPUTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kOutputTag).Set<CPUTensorPtr>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

Status ImageFrameToCPUTensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status ImageFrameToCPUTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                    ExecutionContext* exec_ctx) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  auto mat_view = mediapipe::formats::MatView(&image_frame);
  // TODO(james): we should do these conversions separately.
  cv::Mat float_img;
  mat_view.convertTo(float_img, CV_32F, 1.0 / 255);

  size_t chan_size = static_cast<size_t>(float_img.rows * float_img.cols) * float_img.elemSize1();
  size_t bytes = chan_size * float_img.channels();
  GML_ASSIGN_OR_RETURN(auto tensor, exec_ctx->TensorPool()->GetTensor(bytes));
  GML_RETURN_IF_ERROR(
      tensor->Reshape(TensorShape{1, float_img.channels(), float_img.rows, float_img.cols}));
  tensor->SetDataType(DataType::FLOAT32);

  // TODO(james): This conversion is duplicated with the image_frame_to_cuda_tensor_calculator. We
  // should refactor this.
  cv::Mat channels[3];
  cv::split(float_img, channels);

  for (const auto& [i, chan] : Enumerate(channels)) {
    std::memcpy(static_cast<uint8_t*>(tensor->data()) + i * chan_size, chan.ptr<float>(0),
                chan_size);
  }

  auto packet = mediapipe::MakePacket<CPUTensorPtr>(std::move(tensor));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kOutputTag).AddPacket(std::move(packet));
  return Status::OK();
}

Status ImageFrameToCPUTensorCalculator::CloseImpl(mediapipe::CalculatorContext*,
                                                  ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(ImageFrameToCPUTensorCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
