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

#include "src/gem/calculators/core/image_quality_calculator.h"

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kHistOutputTag = "IMAGE_HIST";
constexpr std::string_view kQualityOutputTag = "IMAGE_QUALITY";

using ImageHistogram = internal::api::core::v1::ImageHistogram;
using ImageQualityMetrics = internal::api::core::v1::ImageQualityMetrics;

absl::Status ImageQualityCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kHistOutputTag).Set<gml::internal::api::core::v1::ImageHistogram>();
  cc->Outputs().Tag(kQualityOutputTag).Set<gml::internal::api::core::v1::ImageQualityMetrics>();
  return absl::OkStatus();
}

Status ImageQualityCalculator::OpenImpl(mediapipe::CalculatorContext*) { return Status::OK(); }

Status ImageQualityCalculator::ProcessImpl(mediapipe::CalculatorContext* cc) {
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();
  cv::Mat input_mat = mediapipe::formats::MatView(&image_frame);

  if (input_mat.channels() != 3) {
    return error::InvalidArgument("Only RGB images are supported. Image with $0 channels passed in",
                                  input_mat.channels());
  }

  cc->Outputs()
      .Tag(kHistOutputTag)
      .AddPacket(mediapipe::MakePacket<ImageHistogram>().At(cc->InputTimestamp()));

  cc->Outputs()
      .Tag(kQualityOutputTag)
      .AddPacket(mediapipe::MakePacket<ImageQualityMetrics>().At(cc->InputTimestamp()));

  return Status::OK();
}

Status ImageQualityCalculator::CloseImpl(mediapipe::CalculatorContext*) { return Status::OK(); }

REGISTER_CALCULATOR(ImageQualityCalculator);

}  // namespace gml::gem::calculators::core
