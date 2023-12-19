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

#include "src/gem/calculators/core/debug_dump_frame_calculator.h"

#include <chrono>

#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/imgcodecs.hpp>

#include "src/common/base/base.h"

constexpr std::string_view kImageTag = "IMAGE_FRAME";
constexpr std::chrono::seconds kDumpPeriod = std::chrono::seconds{60};
constexpr std::string_view kFilenameTemplate = "$0/debug_image_dump_$1.jpg";
constexpr std::string_view kOutPath = "/tmp";

namespace gml::gem::calculators::core {

absl::Status DebugDumpFrameCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageTag).Set<mediapipe::ImageFrame>();
  return absl::OkStatus();
}

absl::Status DebugDumpFrameCalculator::Open(mediapipe::CalculatorContext*) {
  last_dump_ = std::chrono::steady_clock::now();
  first_ = true;
  return absl::OkStatus();
}

absl::Status DebugDumpFrameCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (!first_ && std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                  last_dump_) < kDumpPeriod) {
    return absl::OkStatus();
  }
  first_ = false;
  const auto& image_frame = cc->Inputs().Tag(kImageTag).Get<mediapipe::ImageFrame>();
  const auto& mat = mediapipe::formats::MatView(&image_frame);

  LOG(INFO) << absl::Substitute("Debug frames: format ($0) width ($1) height ($2)",
                                image_frame.Format(), image_frame.Width(), image_frame.Height());

  auto fname = absl::Substitute(kFilenameTemplate, kOutPath, absl::Now());

  cv::imwrite(fname, mat);

  last_dump_ = std::chrono::steady_clock::now();
  return absl::OkStatus();
}

absl::Status DebugDumpFrameCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DebugDumpFrameCalculator);

}  // namespace gml::gem::calculators::core
