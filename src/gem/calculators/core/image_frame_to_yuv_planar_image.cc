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

#include "src/gem/calculators/core/image_frame_to_yuv_planar_image.h"
#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/yuv_image.h>
#include <mediapipe/util/image_frame_util.h>

#include "src/gem/exec/core/planar_image.h"

namespace gml {
namespace gem {
namespace calculators {
namespace core {

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kYUVImageTag = "YUV_IMAGE";

absl::Status ImageFrameToYUVPlanarImage::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kYUVImageTag).Set<std::unique_ptr<exec::core::PlanarImage>>();
  return absl::OkStatus();
}

absl::Status ImageFrameToYUVPlanarImage::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status ImageFrameToYUVPlanarImage::Process(mediapipe::CalculatorContext* cc) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();

  auto yuv_image = std::make_unique<mediapipe::YUVImage>();
  mediapipe::image_frame_util::ImageFrameToYUVImage(image_frame, yuv_image.get());

  GML_ABSL_ASSIGN_OR_RETURN(auto planar,
                            exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                                std::move(yuv_image), exec::core::ImageFormat::YUV_I420));

  auto packet = mediapipe::MakePacket<std::unique_ptr<exec::core::PlanarImage>>(std::move(planar));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kYUVImageTag).AddPacket(std::move(packet));
  return absl::OkStatus();
}

absl::Status ImageFrameToYUVPlanarImage::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ImageFrameToYUVPlanarImage);

}  // namespace core
}  // namespace calculators
}  // namespace gem
}  // namespace gml
