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
