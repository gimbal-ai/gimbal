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

namespace gml::gem::devices::opencv {

/**
 * Base class for OpenCV sources.
 */
class OpenCVSource {
 public:
  virtual ~OpenCVSource() = default;
  virtual cv::Mat ConsumeFrame() = 0;
  virtual int64_t GetLastCaptureUS() = 0;

  bool IsOpened() { return cap_->isOpened(); }
  double GetProperty(int prop_id) { return cap_->get(prop_id); }

 protected:
  std::unique_ptr<cv::VideoCapture> cap_;
};

}  // namespace gml::gem::devices::opencv
