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

#pragma once

#include <chrono>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_format.pb.h>
#include <opencv4/opencv2/opencv.hpp>

#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/plugin/opencv_cam/optionspb/opencv_cam_calculator_options.pb.h"
#include "src/gem/devices/camera/opencv/opencv_source.h"

namespace gml::gem::calculators::opencv_cam {

using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using us_time = std::chrono::time_point<high_resolution_clock, microseconds>;

/**
 * OpenCVCamSourceCalculator Graph API:
 *
 *  Options:
 *    device_filename Filename to open (e.g. /dev/video0)
 *    max_num_frames Maximum number of frames to process (0 for infinite)
 *  Outputs:
 *    mediapipe::ImageFrame
 *    VIDEO_PRESTREAM Optional video header output at Timestamp::PreStream
 *
 */
class OpenCVCamSourceCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::OpenCVCamSourceCalculatorOptions options_;
  std::unique_ptr<devices::opencv::OpenCVSource> source_ = nullptr;
  uint64_t frame_counter_;

  mediapipe::ImageFormat::Format format_;
  cv::ColorConversionCodes color_conversion_;
  int32_t width_;
  int32_t height_;
  double fps_;
  double frame_count_;

  // Metrics.
  opentelemetry::metrics::Gauge<double>* fps_gauge_;
};

}  // namespace gml::gem::calculators::opencv_cam
