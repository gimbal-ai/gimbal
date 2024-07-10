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

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

// This file provides an implementation of PlanarImage for NvBufSurfaceWrapper.

namespace gml::gem::exec::core {

using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

template <>
Status PlanarImageFor<NvBufSurfaceWrapper>::BuildPlanes() {
  constexpr int kNumPlanes = 3;

  const NvBufSurfaceWrapper* image = image_.get();

  planes_.resize(kNumPlanes);

  for (int i = 0; i < kNumPlanes; ++i) {
    planes_[i].data = reinterpret_cast<uint8_t*>(image->surface().mappedAddr.addr[i]);
    planes_[i].row_stride = image->surface().planeParams.pitch[i];

    if (planes_[i].data == nullptr) {
      return error::Internal("Image plane pointer $0 is nullptr.", i);
    }
  }

  return Status::OK();
}

template <>
size_t PlanarImageFor<NvBufSurfaceWrapper>::Width() const {
  return image_->surface().width;
}

template <>
size_t PlanarImageFor<NvBufSurfaceWrapper>::Height() const {
  return image_->surface().height;
}

}  // namespace gml::gem::exec::core
