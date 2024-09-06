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

#include "src/gem/devices/camera/opencv/opencv_cam.h"

namespace gml::gem::devices::opencv {

OpenCVCam::OpenCVCam(const std::string& device_filename, const std::vector<int>& params,
                     int api_preference) {
  cap_ = std::make_unique<cv::VideoCapture>(device_filename, api_preference, params);
}

cv::Mat OpenCVCam::ConsumeFrame() {
  cv::Mat frame;
  cap_->read(frame);
  if (frame.empty()) {
    cap_->read(frame);
  }
  return frame;
}

int64_t OpenCVCam::GetLastCaptureUS() {
  return 1000 * std::llround(cap_->get(cv::CAP_PROP_POS_MSEC));
}

}  // namespace gml::gem::devices::opencv
