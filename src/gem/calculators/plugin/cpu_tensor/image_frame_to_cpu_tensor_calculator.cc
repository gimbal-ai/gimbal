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
  mat_view.convertTo(float_img, CV_32F);

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
