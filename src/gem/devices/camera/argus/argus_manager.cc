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

#include "src/gem/devices/camera/argus/argus_manager.h"

#include <string>

#include <Argus/Argus.h>
#include <sole.hpp>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/gem/devices/camera/argus/argus_cam.h"
#include "src/gem/devices/camera/argus/uuid_utils.h"

namespace gml::gem::devices::argus {

using ArgusCamManagedPtr = ArgusManager::ArgusCamManagedPtr;

ArgusManager& ArgusManager::GetInstance() {
  static ArgusManager manager;
  return manager;
}

ArgusManager::ArgusManager() {
  // Create the CameraProvider object and get the core interface.
  camera_provider_obj_ = Argus::UniqueObj<Argus::CameraProvider>(Argus::CameraProvider::create());
  if (!camera_provider_obj_) {
    LOG(FATAL) << "Failed to create Argus CameraProvider.";
  }
}

std::vector<Argus::ICameraProperties*> ArgusManager::ListCameraProperties() {
  absl::MutexLock lock(&argus_lock_);
  Argus::ICameraProvider* camera_provider =
      Argus::interface_cast<Argus::ICameraProvider>(camera_provider_obj_);

  std::vector<Argus::CameraDevice*> camera_devices;
  camera_provider->getCameraDevices(&camera_devices);
  std::vector<Argus::ICameraProperties*> camera_properties;
  for (auto* camera : camera_devices) {
    camera_properties.emplace_back(Argus::interface_cast<Argus::ICameraProperties>(camera));
  }
  return camera_properties;
}

StatusOr<ArgusCamManagedPtr> ArgusManager::GetCamera(std::string uuid, uint64_t target_frame_rate) {
  absl::MutexLock lock(&argus_lock_);

  if (inuse_ids_.contains(uuid)) {
    return error::InvalidArgument("ArgusCam $0 already in use.", uuid);
  }

  Argus::ICameraProvider* camera_provider =
      Argus::interface_cast<Argus::ICameraProvider>(camera_provider_obj_);

  std::vector<Argus::CameraDevice*> camera_devices;
  camera_provider->getCameraDevices(&camera_devices);

  Argus::UUID argus_uuid = ToArgusUUID(sole::rebuild(uuid));

  Argus::CameraDevice* selected_camera;
  for (auto* camera : camera_devices) {
    Argus::ICameraProperties* camera_properties =
        Argus::interface_cast<Argus::ICameraProperties>(camera);
    if (camera_properties->getUUID() == argus_uuid) {
      selected_camera = camera;
      break;
    }
  }
  if (selected_camera == nullptr) {
    LOG(WARNING) << "Couldn't find argus camera: " << uuid;
    return error::Internal("couldn't find selected camera");
  }

  // Create the capture session.
  auto capture_session = Argus::UniqueObj<Argus::CaptureSession>(
      camera_provider->createCaptureSession(selected_camera));
  if (!capture_session) {
    return error::Internal("failed to create capture session");
  }

  inuse_ids_.insert(uuid);
  return ArgusCamFactory::Create(this, selected_camera, std::move(capture_session),
                                 target_frame_rate, std::move(uuid));
}

void ArgusManager::Release(ArgusCam* cam) {
  absl::MutexLock lock(&argus_lock_);
  inuse_ids_.erase(cam->UUID());
}

ArgusCamManagedPtr ArgusCamFactory::Create(ArgusManager* mgr, Argus::CameraDevice* device,
                                           Argus::UniqueObj<Argus::CaptureSession> session,
                                           uint64_t target_frame_rate, std::string uuid) {
  return ArgusCamManagedPtr(
      new ArgusCam(device, std::move(session), target_frame_rate, std::move(uuid)),
      ArgusManager::Deleter(mgr));
}

}  // namespace gml::gem::devices::argus
