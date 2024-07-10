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
