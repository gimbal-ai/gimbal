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

#include "src/gem/capabilities/plugin/opencv_cam/capability_lister.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>

#include <filesystem>

#include "src/common/system/linux_file_wrapper.h"

namespace gml::gem::capabilities::opencv_cam {

DEFINE_string(video_source, gflags::StringFromEnv("GML_VIDEO_SOURCE", ""),
              "A video source to use as the input instead of a camera attached to the system. For "
              "example a video file or a RSTP stream.");

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
  cap->add_camera_drivers()->set_driver(
      internal::api::core::v1::DeviceCapabilities::CAMERA_DRIVER_V4L2);

  if (FLAGS_video_source != "") {
    auto mutable_cam = cap->add_cameras();
    mutable_cam->set_driver(internal::api::core::v1::DeviceCapabilities::CAMERA_DRIVER_V4L2);

    mutable_cam->set_camera_id(FLAGS_video_source);
    return Status::OK();
  }

  const std::filesystem::path dev{"/dev"};
  std::error_code ec;

  for (auto const& dir_entry : std::filesystem::directory_iterator{dev, ec}) {
    if (!absl::StartsWith(dir_entry.path().string(), "/dev/video")) {
      continue;
    }

    auto file_or_s = system::LinuxFile::Open(dir_entry.path(), O_RDONLY);
    if (!file_or_s.ok()) {
      continue;
    }

    auto file = file_or_s.ConsumeValueOrDie();

    if (file->fd() == -1) {
      continue;
    }

    v4l2_capability c;
    if (ioctl(file->fd(), VIDIOC_QUERYCAP, &c) == -1) {
      continue;
    }

    if (!(c.device_caps & V4L2_CAP_VIDEO_CAPTURE)) {
      continue;
    }

    // Filter for only devices that have mjpeg support.
    // TODO(james/vihang): Push up available video formats to the controlplane and let it decide
    // which ones to use.
    bool hasMJPEG = false;
    v4l2_fmtdesc fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.index = 0;
    while (ioctl(file->fd(), VIDIOC_ENUM_FMT, &fmt) != -1) {
      if (fmt.pixelformat == v4l2_fourcc('M', 'J', 'P', 'G')) {
        hasMJPEG = true;
        break;
      }
      fmt.index++;
    }
    if (!hasMJPEG) {
      continue;
    }

    auto mutable_cam = cap->add_cameras();
    mutable_cam->set_driver(internal::api::core::v1::DeviceCapabilities::CAMERA_DRIVER_V4L2);

    mutable_cam->set_camera_id(dir_entry.path());
  }

  if (ec) {
    return error::Internal("Error when iterating dir: $0", ec.message());
  }

  return Status::OK();
}

}  // namespace gml::gem::capabilities::opencv_cam
