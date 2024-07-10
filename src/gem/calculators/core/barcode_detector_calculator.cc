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

#include "src/gem/calculators/core/barcode_detector_calculator.h"

#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/barcode.hpp>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

constexpr std::string_view kImageTag = "IMAGE_FRAME";
constexpr std::string_view kDetectionsTag = "DETECTIONS";

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

absl::Status BarcodeDetectorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status BarcodeDetectorCalculator::Open(mediapipe::CalculatorContext*) {
  detector_ = std::make_unique<cv::barcode::BarcodeDetector>();
  return absl::OkStatus();
}

absl::Status BarcodeDetectorCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& image_frame = cc->Inputs().Tag(kImageTag).Get<mediapipe::ImageFrame>();
  const auto& mat = mediapipe::formats::MatView(&image_frame);

  const auto width = static_cast<float>(image_frame.Width());
  const auto height = static_cast<float>(image_frame.Height());

  cv::Mat grayscale;
  // TODO(james): Investigate if this improves the results.
  cv::cvtColor(mat, grayscale, cv::COLOR_BGR2GRAY);

  std::vector<cv::Point2f> corners;
  std::vector<std::string> decoded_info;
  std::vector<std::string> decoded_type;
  detector_->detectAndDecodeWithType(grayscale, decoded_info, decoded_type, corners);

  std::vector<Detection> detections;
  for (const auto& [detect_idx, info] : Enumerate(decoded_info)) {
    auto& d = detections.emplace_back();

    auto type = decoded_type[detect_idx];
    auto* label = d.add_label();
    // TODO(james): not sure what score should be.
    label->set_score(1.0);
    if (info == "") {
      label->set_label(absl::Substitute("Barcode: <decode failed>"));
    } else {
      label->set_label(absl::Substitute("Barcode: [$0] $1", type, info));
    }

    // We build a non-rotated bounding box based on the (potentially) rotated rectangle opencv
    // returns.
    float min_x = width;
    float max_x = 0.0;
    float min_y = height;
    float max_y = 0.0;
    for (int i = 0; i < 4; ++i) {
      const auto& point = corners[4 * detect_idx + i];
      if (point.x < min_x) {
        min_x = point.x;
      }
      if (point.x > max_x) {
        max_x = point.x;
      }
      if (point.y < min_y) {
        min_y = point.y;
      }
      if (point.y > max_y) {
        max_y = point.y;
      }
    }
    auto xc = (min_x + max_x) / 2;
    xc = xc / width;
    auto yc = (min_y + max_y) / 2;
    yc = yc / height;

    auto bbox_width = (max_x - min_x) / width;
    auto bbox_height = (max_y - min_y) / height;

    auto bbox = d.mutable_bounding_box();
    bbox->set_xc(xc);
    bbox->set_yc(yc);
    bbox->set_width(bbox_width);
    bbox->set_height(bbox_height);
  }

  cc->Outputs()
      .Tag(kDetectionsTag)
      .AddPacket(mediapipe::MakePacket<std::vector<Detection>>(std::move(detections))
                     .At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status BarcodeDetectorCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(BarcodeDetectorCalculator);

}  // namespace gml::gem::calculators::core
