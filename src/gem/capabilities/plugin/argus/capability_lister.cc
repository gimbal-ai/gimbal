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

#include "src/gem/capabilities/plugin/argus/capability_lister.h"

#include <Argus/Argus.h>

#include "src/gem/devices/camera/argus/argus_manager.h"
#include "src/gem/devices/camera/argus/uuid_utils.h"

namespace gml::gem::capabilities::argus {

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
  auto& argus_manager = devices::argus::ArgusManager::GetInstance();

  for (const auto& props : argus_manager.ListCameraProperties()) {
    auto mutable_cam = cap->add_cameras();
    mutable_cam->set_driver(
        internal::api::core::v1::DeviceCapabilities::CameraInfo::CAMERA_DRIVER_ARGUS);

    mutable_cam->set_camera_id(devices::argus::ParseUUID(props->getUUID()).str());
  }

  return Status::OK();
}

}  // namespace gml::gem::capabilities::argus
