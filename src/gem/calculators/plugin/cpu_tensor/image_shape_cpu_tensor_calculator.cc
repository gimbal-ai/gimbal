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

#include "src/gem/calculators/plugin/cpu_tensor/image_shape_cpu_tensor_calculator.h"

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
constexpr std::string_view kOutputTag = "IMAGE_SHAPE";

absl::Status ImageShapeCPUTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kOutputTag).Set<CPUTensorPtr>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

Status ImageShapeCPUTensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status ImageShapeCPUTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                  ExecutionContext* exec_ctx) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  GML_ASSIGN_OR_RETURN(auto cpu_tensor, exec_ctx->TensorPool()->GetTensor(sizeof(float) * 2));
  GML_RETURN_IF_ERROR(cpu_tensor->Reshape(TensorShape{2}));
  cpu_tensor->SetDataType(DataType::INT32);

  int32_t* shape = cpu_tensor->TypedData<DataType::INT32>();

  shape[0] = image_frame.Width();
  shape[1] = image_frame.Height();

  auto packet = mediapipe::MakePacket<CPUTensorPtr>(std::move(cpu_tensor));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kOutputTag).AddPacket(std::move(packet));
  return Status::OK();
}

Status ImageShapeCPUTensorCalculator::CloseImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(ImageShapeCPUTensorCalculator);

}  // namespace gml::gem::calculators::cpu_tensor
