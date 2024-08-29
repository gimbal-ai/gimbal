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

#include "src/gem/calculators/core/image_frame_to_yuv_planar_image.h"

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/yuv_image.h>
#include <mediapipe/util/image_frame_util.h>

#include "src/gem/exec/core/planar_image.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kYUVImageTag = "YUV_IMAGE";

absl::Status ImageFrameToYUVPlanarImage::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kYUVImageTag).Set<std::unique_ptr<exec::core::PlanarImage>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ImageFrameToYUVPlanarImage::Process(mediapipe::CalculatorContext* cc) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  auto yuv_image = std::make_shared<mediapipe::YUVImage>();
  mediapipe::image_frame_util::ImageFrameToYUVImage(image_frame, yuv_image.get());

  GML_ABSL_ASSIGN_OR_RETURN(auto planar,
                            exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                                std::move(yuv_image), exec::core::ImageFormat::YUV_I420));

  auto packet = mediapipe::MakePacket<std::unique_ptr<exec::core::PlanarImage>>(std::move(planar));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kYUVImageTag).AddPacket(std::move(packet));
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ImageFrameToYUVPlanarImage);

}  // namespace gml::gem::calculators::core
