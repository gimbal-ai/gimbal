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

#include "src/gem/calculators/plugin/opencv_cam/opencv_cam_calculator.h"

#include <magic_enum.hpp>

#include "absl/status/status.h"

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>

namespace gml {
namespace gem {
namespace calculators {
namespace opencv {

// using ::gml::gem::calculators::argus::optionspb::OpenCVCamSourceCalculatorOptions;

namespace {

absl::Status SetupFormatConversion(const cv::Mat& frame, mediapipe::ImageFormat::Format* format_out,
                                   cv::ColorConversionCodes* color_conversion_out) {
  auto& format = *format_out;
  auto& color_conversion = *color_conversion_out;

  switch (frame.channels()) {
    case 1:
      format = mediapipe::ImageFormat::GRAY8;
      return absl::UnavailableError(
          absl::Substitute("Unsupported format $0", magic_enum::enum_name(format)));
    case 3:
      format = mediapipe::ImageFormat::SRGB;
      color_conversion = cv::COLOR_BGR2RGB;
      return absl::OkStatus();
    case 4:
      format = mediapipe::ImageFormat::SRGBA;
      color_conversion = cv::COLOR_BGRA2RGBA;
      return absl::OkStatus();
    default:
      break;
  }

  return absl::UnavailableError(
      absl::Substitute("Unsupported format [num channels = $0]", frame.channels()));
}
}  // namespace

cv::Mat OpenCVCamSourceCalculator::CaptureFrame() {
  cv::Mat frame;

  // Capture a single frame
  cap_->read(frame);

  // Apparently, the first read can fail sometimes, so try a second time just to be sure.
  // This was noted from a comment in the  Mediapipe source code.
  if (frame.empty()) {
    cap_->read(frame);  // Try again.
  }

  return frame;
}

absl::Status OpenCVCamSourceCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Outputs().Index(0).Set<mediapipe::ImageFrame>();
  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<OpenCVCamSourceCalculatorOptions>();

  // Setting to use V4L2 only. Could revisit this choice in the future.
  cap_ = std::make_unique<cv::VideoCapture>(options_.device_num(), cv::CAP_V4L2);

  if (!cap_->isOpened()) {
    return absl::InternalError("Could not open camera.");
  }

  // Grab a frame to see its parameters;
  cv::Mat frame = CaptureFrame();
  if (frame.empty()) {
    return absl::InternalError("Could not capture image with camera.");
  }

  width_ = static_cast<int32_t>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int32_t>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
  MP_RETURN_IF_ERROR(SetupFormatConversion(frame, &format_, &color_conversion_));

  timestamp_ = 0;
  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Process(mediapipe::CalculatorContext* cc) {
  absl::Status s;

  cv::Mat frame = CaptureFrame();

  if (frame.empty()) {
    return mediapipe::tool::StatusStop();
  }

  // Convert-copy the cv::Mat into a mediapipe::ImageFormat.
  constexpr int kAlignmentBoundary = 1;
  auto image_frame =
      absl::make_unique<mediapipe::ImageFrame>(format_, width_, height_, kAlignmentBoundary);
  cv::cvtColor(frame, mediapipe::formats::MatView(image_frame.get()), color_conversion_);

  auto packet = mediapipe::Adopt(image_frame.release()).At(mediapipe::Timestamp(timestamp_));
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  ++timestamp_;

  if (options_.max_num_frames() > 0 && timestamp_ >= options_.max_num_frames()) {
    return mediapipe::tool::StatusStop();
  }

  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  cap_->release();
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCVCamSourceCalculator);

}  // namespace opencv
}  // namespace calculators
}  // namespace gem
}  // namespace gml