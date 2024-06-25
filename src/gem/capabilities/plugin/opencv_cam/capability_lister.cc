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

#include "src/gem/capabilities/plugin/opencv_cam/capability_lister.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>

#include <filesystem>

#include "src/common/system/linux_file_wrapper.h"

namespace gml::gem::capabilities::opencv_cam {

DEFINE_string(video_from_file_override, gflags::StringFromEnv("GML_VIDEO_FROM_FILE_OVERRIDE", ""),
              "A video file to use as the input instead of a camera attached to the system.");

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
  if (FLAGS_video_from_file_override != "") {
    auto mutable_cam = cap->add_cameras();
    mutable_cam->set_driver(
        internal::api::core::v1::DeviceCapabilities::CameraInfo::CAMERA_DRIVER_V4L2);

    mutable_cam->set_camera_id(FLAGS_video_from_file_override);
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
    mutable_cam->set_driver(
        internal::api::core::v1::DeviceCapabilities::CameraInfo::CAMERA_DRIVER_V4L2);

    mutable_cam->set_camera_id(dir_entry.path());
  }

  if (ec) {
    return error::Internal("Error when iterating dir: $0", ec.message());
  }

  return Status::OK();
}

}  // namespace gml::gem::capabilities::opencv_cam
