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

#include "src/gem/calculators/plugin/opencv_cam/opencv_cam_calculator.h"

#include <thread>
#include <vector>

#include <absl/status/status.h>
#include <magic_enum.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/formats/video_stream_header.h>
#include <mediapipe/framework/timestamp.h>

#include "src/gem/devices/camera/opencv/opencv_cam.h"
#include "src/gem/devices/camera/opencv/opencv_video.h"

namespace gml::gem::calculators::opencv_cam {

constexpr std::string_view kVideoPrestreamTag = "VIDEO_PRESTREAM";

constexpr int kTargetFrameWidth = 800;
constexpr int kTargetFrameHeight = 600;
constexpr int kTargetFPS = 30;

const int kTargetFourCC = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

using ::gml::gem::calculators::opencv_cam::optionspb::OpenCVCamSourceCalculatorOptions;

namespace {

absl::Status SetupFormatConversion(const cv::Mat& frame, mediapipe::ImageFormat::Format* format_out,
                                   cv::ColorConversionCodes* color_conversion_out) {
  auto& format = *format_out;
  auto& color_conversion = *color_conversion_out;

  switch (frame.channels()) {
    case 1:
      format = mediapipe::ImageFormat::FORMAT_GRAY8;
      return absl::UnavailableError(
          absl::Substitute("Unsupported format $0", magic_enum::enum_name(format)));
    case 3:
      VLOG(1) << "Source has BGR format";
      format = mediapipe::ImageFormat::FORMAT_SRGB;
      color_conversion = cv::COLOR_BGR2RGB;
      return absl::OkStatus();
    case 4:
      VLOG(1) << "Source has BGRA format";
      format = mediapipe::ImageFormat::FORMAT_SRGBA;
      color_conversion = cv::COLOR_BGRA2RGBA;
      return absl::OkStatus();
    default:
      break;
  }

  return absl::UnavailableError(
      absl::Substitute("Unsupported format [num channels = $0]", frame.channels()));
}
}  // namespace

absl::Status OpenCVCamSourceCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Outputs().Index(0).Set<mediapipe::ImageFrame>();
  if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
    cc->Outputs().Tag(kVideoPrestreamTag).Set<mediapipe::VideoHeader>();
  }
  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<OpenCVCamSourceCalculatorOptions>();

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  fps_gauge_ = metrics_system.GetOrCreateGauge<double>("gml.gem.camera.fps", "Camera FPS");

  LOG(INFO) << "Using v4l2 camera: " << options_.device_filename();

  if (absl::StartsWith(options_.device_filename(), "/dev/video")) {
    std::vector<int> params = {
        cv::CAP_PROP_FRAME_WIDTH, kTargetFrameWidth, cv::CAP_PROP_FRAME_HEIGHT, kTargetFrameHeight,
        cv::CAP_PROP_FOURCC,      kTargetFourCC,     cv::CAP_PROP_FPS,          kTargetFPS,
    };
    source_ = std::make_unique<devices::opencv::OpenCVCam>(options_.device_filename(), params);
  } else if (absl::StrContains(options_.device_filename(), "://")) {
    std::vector<int> params = {};
    source_ = std::make_unique<devices::opencv::OpenCVCam>(options_.device_filename(), params,
                                                           cv::CAP_FFMPEG);
  } else {
    std::vector<int> params = {};
    source_ = std::make_unique<devices::opencv::OpenCVVideo>(options_.device_filename(), params);
  }

  if (!source_->IsOpened()) {
    return absl::InternalError(
        absl::Substitute("Failed to open camera source: $0", options_.device_filename()));
  }

  width_ = static_cast<int32_t>(source_->GetProperty(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int32_t>(source_->GetProperty(cv::CAP_PROP_FRAME_HEIGHT));
  fps_ = source_->GetProperty(cv::CAP_PROP_FPS);

  auto frame = source_->ConsumeFrame();
  if (frame.empty()) {
    LOG(WARNING) << "Failed to capture image with: " << options_.device_filename();
    return absl::InternalError("Could not capture image with camera.");
  }

  MP_RETURN_IF_ERROR(SetupFormatConversion(frame, &format_, &color_conversion_));

  auto header = std::make_unique<mediapipe::VideoHeader>();
  header->format = format_;
  header->width = width_;
  header->height = height_;

  LOG(INFO) << "Video header: " << header->width << "x" << header->height;

  if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
    cc->Outputs().Tag(kVideoPrestreamTag).Add(header.release(), mediapipe::Timestamp::PreStream());
    cc->Outputs().Tag(kVideoPrestreamTag).Close();
  }

  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto frame = source_->ConsumeFrame();
  if (frame.empty()) {
    return mediapipe::tool::StatusStop();
  }

  int64_t timestamp = source_->GetLastCaptureUS();

  // Convert-copy the cv::Mat into a mediapipe::ImageFormat.
  constexpr int kAlignmentBoundary = 1;
  auto image_frame =
      absl::make_unique<mediapipe::ImageFrame>(format_, width_, height_, kAlignmentBoundary);
  cv::cvtColor(frame, mediapipe::formats::MatView(image_frame.get()), color_conversion_);

  auto packet = mediapipe::Adopt(image_frame.release()).At(mediapipe::Timestamp(timestamp));
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  fps_gauge_->Record(fps_, {{"camera_id", options_.device_filename()}, {"camera", "opencv"}}, {});

  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCVCamSourceCalculator);

}  // namespace gml::gem::calculators::opencv_cam
