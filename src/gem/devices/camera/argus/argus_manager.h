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

#pragma once

#include <memory>

#include <Argus/Argus.h>
#include <absl/container/flat_hash_set.h>

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/argus_cam.h"

namespace gml::gem::devices::argus {

/**
 * Singleton that manages access to Argus cameras.
 *
 * Argus seems to behave poorly when multiple threads try to use the Argus API simultaneously.
 */
class ArgusManager {
  static constexpr uint64_t kDefaultTargetFrameRate = 30;

 public:
  static ArgusManager& GetInstance();

  class Deleter {
   public:
    Deleter() = default;
    explicit Deleter(ArgusManager* mgr) : mgr_(mgr) {}
    void operator()(ArgusCam* cam) {
      if (mgr_ != nullptr) {
        mgr_->Release(cam);
      }
      std::default_delete<ArgusCam>()(cam);
    }

   private:
    ArgusManager* mgr_;
  };
  using ArgusCamManagedPtr = std::unique_ptr<ArgusCam, Deleter>;

  std::vector<Argus::ICameraProperties*> ListCameraProperties() ABSL_LOCKS_EXCLUDED(argus_lock_);
  StatusOr<ArgusCamManagedPtr> GetCamera(std::string uuid,
                                         uint64_t target_frame_rate = kDefaultTargetFrameRate)
      ABSL_LOCKS_EXCLUDED(argus_lock_);

  ArgusManager(const ArgusManager&) = delete;
  void operator=(const ArgusManager&) = delete;

 private:
  ArgusManager();
  void Release(ArgusCam*);

  absl::Mutex argus_lock_;
  Argus::UniqueObj<Argus::CameraProvider> camera_provider_obj_ ABSL_GUARDED_BY(argus_lock_);
  absl::flat_hash_set<std::string> inuse_ids_ ABSL_GUARDED_BY(argus_lock_);
};

/**
 * ArgusCamFactory ensures that only ArgusManager can create an ArgusCam.
 * This uses the so-called Attorney-Client pattern, to avoid giving ArgusManager full friend
 * access to ArgusCam, while still ensuring that only ArgusManager can construct an ArgusCam.
 */
class ArgusCamFactory {
 private:
  static ArgusManager::ArgusCamManagedPtr Create(ArgusManager* mgr, Argus::CameraDevice* device,
                                                 Argus::UniqueObj<Argus::CaptureSession> session,
                                                 uint64_t target_frame_rate, std::string uuid);
  friend class ArgusManager;
};

}  // namespace gml::gem::devices::argus
