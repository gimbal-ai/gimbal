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

#include "src/gem/devices/camera/opencv/opencv_video.h"

#include <chrono>

namespace gml::gem::devices::opencv {

OpenCVVideo::OpenCVVideo(const std::string& device_filename, const std::vector<int>& params,
                         int api_preference) {
  cap_ = std::make_unique<cv::VideoCapture>(device_filename, api_preference, params);
  frame_count_ = cap_->get(cv::CAP_PROP_FRAME_COUNT);
  video_start_ms_ = std::llround(cap_->get(cv::CAP_PROP_POS_MSEC));
  // This offsets the recorded videos into "real time".
  video_start_offset_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::high_resolution_clock::now().time_since_epoch())
                               .count();
}

cv::Mat OpenCVVideo::NextFrame() {
  cv::Mat frame;
  cap_->read(frame);
  if (frame.empty()) {
    cap_->read(frame);
  }
  return frame;
}

cv::Mat OpenCVVideo::ConsumeFrame() {
  bool looped = false;
  int64_t loop_sleep_ms = 0;
  auto curr_frame = cap_->get(cv::CAP_PROP_POS_FRAMES);
  if (curr_frame >= frame_count_) {
    auto end_ms = std::llround(cap_->get(cv::CAP_PROP_POS_MSEC));

    // Ususally we sleep for the time between frames, but if we looped back to the start, we don't
    // have a delta, so sleep for the inverse of the fps (and add that to the video length)
    auto fps = cap_->get(cv::CAP_PROP_FPS);
    loop_sleep_ms = std::llround(1000 / fps);
    video_length_ms_ = end_ms - video_start_ms_ + loop_sleep_ms;

    looped = true;
    cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
    loop_count_++;
  }

  // We must read the frame before reading the position, this ensures that the possible double reads
  // needed at the start of the video are handled correctly.
  auto frame = NextFrame();
  auto curr_ts_ms = std::llround(cap_->get(cv::CAP_PROP_POS_MSEC));

  if (looped) {
    std::this_thread::sleep_for(std::chrono::milliseconds{loop_sleep_ms});
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds{curr_ts_ms - last_frame_timestamp_ms_});
  }
  last_frame_timestamp_ms_ = curr_ts_ms;

  return frame;
}

int64_t OpenCVVideo::GetLastCaptureUS() {
  return video_start_offset_us_ +
         1000 * ((video_length_ms_ * loop_count_) + (last_frame_timestamp_ms_ - video_start_ms_));
}

}  // namespace gml::gem::devices::opencv
