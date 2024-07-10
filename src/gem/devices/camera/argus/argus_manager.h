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
