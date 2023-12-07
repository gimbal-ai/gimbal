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

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <filesystem>

#include "src/common/system/linux_file_wrapper.h"

#include "src/gem/capabilities/plugin/opencv_cam/capability_lister.h"

namespace gml::gem::capabilities::opencv_cam {

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
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

    if (!(c.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
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
