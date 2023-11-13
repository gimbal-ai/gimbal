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

#include <cuda_runtime_api.h>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/tensorrt/image_frame_to_cuda_tensor_calculator.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml::gem::calculators::tensorrt {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::tensorrt::CUDATensorPtr;

static constexpr char kImageFrameTag[] = "IMAGE_FRAME";
static constexpr char kTensorTag[] = "OUTPUT_TENSOR";

absl::Status ImageFrameToCUDATensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kTensorTag).Set<CUDATensorPtr>();
  return absl::OkStatus();
}

Status ImageFrameToCUDATensorCalculator::OpenImpl(mediapipe::CalculatorContext*,
                                                  ExecutionContext*) {
  return Status::OK();
}

Status ImageFrameToCUDATensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                     ExecutionContext* exec_ctx) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  if (image_frame.NumberOfChannels() != 3) {
    return Status(types::CODE_INVALID_ARGUMENT, "Only 3 channel images are currently supported");
  }
  auto mat_view = mediapipe::formats::MatView(&image_frame);
  if (!mat_view.isContinuous()) {
    return Status(types::CODE_INVALID_ARGUMENT,
                  "Only contiguous opencv matrices are supported for now");
  }

  // TODO(james): currently only supporting 3 channel RGB images with values in 0-255.
  if (mat_view.type() != CV_8UC3) {
    return Status(types::CODE_INVALID_ARGUMENT,
                  "Only 3 channel unsigned char images are currently supported.");
  }

  // TODO(james): we should do these conversions separately.
  cv::Mat float_img;
  mat_view.convertTo(float_img, CV_32F, 1.0 / 255);

  size_t chan_size = float_img.rows * float_img.cols * float_img.elemSize1();
  size_t bytes = chan_size * float_img.channels();
  GML_ASSIGN_OR_RETURN(auto tensor, exec_ctx->TensorPool()->GetTensor(bytes));

  GML_RETURN_IF_ERROR(
      tensor->Reshape(TensorShape{1, float_img.channels(), float_img.rows, float_img.cols}));
  tensor->SetDataType(DataType::FLOAT32);

  // TODO(james): This is inefficient. We should convert directly to CUDA NHWC and then use the GPU
  // to convert to NCHW.
  cv::Mat channels[3];
  cv::split(float_img, channels);

  for (const auto& [i, chan] : Enumerate(channels)) {
    if (cudaMemcpy(static_cast<uint8_t*>(tensor->data()) + i * chan_size, chan.ptr<float>(0),
                   chan_size, cudaMemcpyHostToDevice) != cudaSuccess) {
      return Status(types::CODE_INTERNAL, "Failed to memcpy from host to cuda device");
    }
  }

  auto packet = mediapipe::MakePacket<CUDATensorPtr>(std::move(tensor));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kTensorTag).AddPacket(std::move(packet));
  return Status::OK();
}
Status ImageFrameToCUDATensorCalculator::CloseImpl(mediapipe::CalculatorContext*,
                                                   ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(ImageFrameToCUDATensorCalculator);

}  // namespace gml::gem::calculators::tensorrt
