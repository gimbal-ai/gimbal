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

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

// This file provides an implementation of PlanarImage for NvBufSurfaceWrapper.

namespace gml {
namespace gem {
namespace exec {
namespace core {

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

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
