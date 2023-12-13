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

#include "src/gem/capabilities/plugin/argus/uuid_utils.h"

namespace gml::gem::capabilities::argus {

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
  auto camera_provider_obj =
      Argus::UniqueObj<Argus::CameraProvider>(Argus::CameraProvider::create());
  Argus::ICameraProvider* camera_provider =
      Argus::interface_cast<Argus::ICameraProvider>(camera_provider_obj);
  if (camera_provider == nullptr) {
    return error::Internal("Failed to create CameraProvider.");
  }

  std::vector<Argus::CameraDevice*> camera_devices;
  camera_provider->getCameraDevices(&camera_devices);

  for (const auto& camera_device : camera_devices) {
    // TODO(vihang): Grab other camera info from the device.
    Argus::ICameraProperties* camera_properties =
        Argus::interface_cast<Argus::ICameraProperties>(camera_device);

    auto mutable_cam = cap->add_cameras();
    mutable_cam->set_driver(
        internal::api::core::v1::DeviceCapabilities::CameraInfo::CAMERA_DRIVER_ARGUS);

    mutable_cam->set_camera_id(
        gem::capabilities::argus::ParseUUID(camera_properties->getUUID()).str());
  }

  return Status::OK();
}

}  // namespace gml::gem::capabilities::argus
