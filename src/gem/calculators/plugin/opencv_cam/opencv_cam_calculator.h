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

#pragma once

#include <chrono>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_format.pb.h>
#include <opencv4/opencv2/opencv.hpp>

#include "src/gem/calculators/plugin/opencv_cam/optionspb/opencv_cam_calculator_options.pb.h"

namespace gml::gem::calculators::opencv_cam {

using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using us_time = std::chrono::time_point<high_resolution_clock, microseconds>;

class OpenCVCamSourceCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  cv::Mat LoopCapture();
  cv::Mat Read();

  optionspb::OpenCVCamSourceCalculatorOptions options_;
  std::unique_ptr<cv::VideoCapture> cap_;
  uint64_t frame_count_;

  // Only used for when the source is a video file and not a device.
  int64_t video_start_offset_us_;

  mediapipe::ImageFormat::Format format_;
  cv::ColorConversionCodes color_conversion_;
  int32_t width_;
  int32_t height_;
};

}  // namespace gml::gem::calculators::opencv_cam
