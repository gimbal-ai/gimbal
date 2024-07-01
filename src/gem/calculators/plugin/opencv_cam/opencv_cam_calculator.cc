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

#include <thread>
#include <vector>

#include <absl/status/status.h>
#include <magic_enum.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/formats/video_stream_header.h>
#include <mediapipe/framework/timestamp.h>

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

cv::Mat OpenCVCamSourceCalculator::LoopCapture() {
  cv::Mat frame;

  auto curr_frame = cap_->get(cv::CAP_PROP_POS_FRAMES);
  if (frame_count_ <= 0 || curr_frame == -1) {
    // This is a camera, not a video.
    cap_->read(frame);
    return frame;
  }

  // If we are here, this is a video file.
  if (video_start_offset_us_ == 0) {
    // The video file stream starts at time pos 0. This offset indicated the delta between the file
    // and current time. Used to both adjust the mediapipe output ts and to throttle the video read
    // from file to match the expected recorded FPS.
    video_start_offset_us_ =
        std::chrono::duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch())
            .count();
  }

  if (curr_frame >= frame_count_) {
    // Reset to beginning of video.

    // Push offset by length of video. (add one for the gap between the last frame and looped zeroth
    // frame).
    video_start_offset_us_ += std::lround((frame_count_ + 1) * 1000 * 1000 / fps_);
    // Reset current frame.
    cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
  }

  // 0 indexed timestamp of the current frame.
  auto curr_ts_ms = cap_->get(cv::CAP_PROP_POS_MSEC);

  int64_t now_us =
      std::chrono::duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch())
          .count();
  int64_t expected_output_us = video_start_offset_us_ + std::llround(curr_ts_ms * 1000);
  int64_t delta_us = expected_output_us - now_us;

  if (delta_us > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds{delta_us});
  }

  cap_->read(frame);
  return frame;
}

cv::Mat OpenCVCamSourceCalculator::Read() {
  cv::Mat frame = LoopCapture();
  // Apparently, the first read can fail sometimes, so try a second time just to be sure.
  // This was noted from a comment in the  Mediapipe source code.
  if (frame.empty()) {
    frame = LoopCapture();  // Try again.
  }

  return frame;
}

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
  auto gml_meter = metrics_system.GetMeterProvider()->GetMeter("gml");
  fps_gauge_ = gml_meter->CreateDoubleGauge("gml.gem.camera.fps");

  LOG(INFO) << "Using v4l2 camera: " << options_.device_filename();

  std::vector<int> params;
  if (absl::StartsWith(options_.device_filename(), "/dev/video")) {
    params.insert(params.end(), {
                                    cv::CAP_PROP_FRAME_WIDTH,
                                    kTargetFrameWidth,
                                    cv::CAP_PROP_FRAME_HEIGHT,
                                    kTargetFrameHeight,
                                    cv::CAP_PROP_FOURCC,
                                    kTargetFourCC,
                                });
  }

  cap_ = std::make_unique<cv::VideoCapture>(options_.device_filename(), cv::CAP_ANY, params);
  if (absl::StartsWith(options_.device_filename(), "/dev/video")) {
    cap_->set(cv::CAP_PROP_FPS, kTargetFPS);
  }

  if (!cap_->isOpened()) {
    LOG(WARNING) << "Failed to open camera: " << options_.device_filename();
    return absl::InternalError("Could not open camera.");
  }

  width_ = static_cast<int32_t>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int32_t>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
  // fps and frame_count must be set before calling Read.
  fps_ = cap_->get(cv::CAP_PROP_FPS);
  frame_count_ = cap_->get(cv::CAP_PROP_FRAME_COUNT);

  // Grab a frame to see its parameters;
  cv::Mat frame = Read();
  if (frame.empty()) {
    LOG(WARNING) << "Failed to capture image with: " << options_.device_filename();
    return absl::InternalError("Could not capture image with camera.");
  }

  MP_RETURN_IF_ERROR(SetupFormatConversion(frame, &format_, &color_conversion_));

  auto header = std::make_unique<mediapipe::VideoHeader>();
  header->format = format_;
  header->width = width_;
  header->height = height_;
  header->frame_rate = fps_;
  header->duration = static_cast<float>(frame_count_ / fps_);

  LOG(INFO) << "Video header: " << header->width << "x" << header->height << " @ "
            << header->frame_rate << " fps";

  if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
    cc->Outputs().Tag(kVideoPrestreamTag).Add(header.release(), mediapipe::Timestamp::PreStream());
    cc->Outputs().Tag(kVideoPrestreamTag).Close();
  }

  frame_counter_ = 0;
  video_start_offset_us_ = 0;
  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Process(mediapipe::CalculatorContext* cc) {
  absl::Status s;

  cv::Mat frame = Read();

  if (frame.empty()) {
    return mediapipe::tool::StatusStop();
  }

  int64_t timestamp = std::llround(cap_->get(cv::CAP_PROP_POS_MSEC) * 1000);

  // Convert-copy the cv::Mat into a mediapipe::ImageFormat.
  constexpr int kAlignmentBoundary = 1;
  auto image_frame =
      absl::make_unique<mediapipe::ImageFrame>(format_, width_, height_, kAlignmentBoundary);
  cv::cvtColor(frame, mediapipe::formats::MatView(image_frame.get()), color_conversion_);

  auto packet = mediapipe::Adopt(image_frame.release())
                    .At(mediapipe::Timestamp(timestamp + video_start_offset_us_));
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  if (options_.max_num_frames() > 0 && ++frame_counter_ >= options_.max_num_frames()) {
    return mediapipe::tool::StatusStop();
  }

  fps_gauge_->Record(fps_, {{"camera_id", options_.device_filename()}, {"camera", "opencv"}}, {});

  return absl::OkStatus();
}

absl::Status OpenCVCamSourceCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  cap_->release();
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCVCamSourceCalculator);

}  // namespace gml::gem::calculators::opencv_cam
