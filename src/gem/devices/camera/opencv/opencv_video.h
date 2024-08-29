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

#include <vector>

#include <opencv4/opencv2/opencv.hpp>

#include "src/common/base/base.h"
#include "src/gem/devices/camera/opencv/opencv_source.h"

namespace gml::gem::devices::opencv {

/**
 * Provides a simple access model to a video using opencv+ffmpeg.
 */
class OpenCVVideo : public OpenCVSource {
 public:
  OpenCVVideo(const std::string& device_filename, const std::vector<int>& params,
              int api_preference = cv::CAP_FFMPEG);
  ~OpenCVVideo() override = default;

  cv::Mat ConsumeFrame() override;

  double GetProperty(int prop_id) override;
  int64_t GetLastCaptureUS() override;

 private:
  cv::Mat NextFrame();

  std::unique_ptr<cv::VideoCapture> cap_;

  double frame_count_;
  int64_t loop_count_ = 0;
  int64_t video_start_ms_ = 0;
  int64_t video_length_ms_ = -1;

  int64_t last_frame_timestamp_ms_ = 0;
  int64_t video_start_offset_us_ = 0;
};

}  // namespace gml::gem::devices::opencv
