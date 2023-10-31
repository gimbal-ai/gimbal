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

#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

#include <iostream>
#include <string>

#include <nvbufsurface.h>

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace devices {
namespace argus {

StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> NvBufSurfaceWrapper::Create(
    NvBufSurface* nvbuf_surf) {
  return absl::WrapUnique<NvBufSurfaceWrapper>(new NvBufSurfaceWrapper(nvbuf_surf));
}

NvBufSurfaceWrapper::NvBufSurfaceWrapper(NvBufSurface* nvbuf_surf) : nvbuf_surf_(nvbuf_surf) {}

NvBufSurfaceWrapper::NvBufSurfaceWrapper(NvBufSurfaceWrapper&& other)
    : nvbuf_surf_(other.nvbuf_surf_) {
  // Moved ownership over, so set pointer to nullptr to ensure there is no deallocation on
  // destructor.
  other.nvbuf_surf_ = nullptr;
}

NvBufSurfaceWrapper::~NvBufSurfaceWrapper() {
  if (nvbuf_surf_ != nullptr) {
    NvBufSurfaceDestroy(nvbuf_surf_);
  }
}

void NvBufSurfaceWrapper::DumpInfo() const {
  LOG(INFO) << absl::Substitute("nvbuf_surf: batchSize $0", nvbuf_surf_->batchSize);
  LOG(INFO) << absl::Substitute("nvbuf_surf: numFilled $0", nvbuf_surf_->numFilled);

  // Only supporting the first buffer for now.
  const NvBufSurfaceParams& surf_params = nvbuf_surf_->surfaceList[0];

  LOG(INFO) << absl::Substitute(
      "surf_params[0]: WxH=$0x$1 pitch=$2 color_format=$3 layout=$4 dataSize=$5 num_planes=$6",
      surf_params.width, surf_params.height, surf_params.pitch,
      magic_enum::enum_name(surf_params.colorFormat), magic_enum::enum_name(surf_params.layout),
      surf_params.dataSize, surf_params.planeParams.num_planes);
  for (uint32_t i = 0; i < surf_params.planeParams.num_planes; ++i) {
    const auto& plane_params = surf_params.planeParams;
    LOG(INFO) << absl::Substitute(
        "plane_params[$0] WxH=$1x$2 pitch=$3 offset=$4 psize=$5 bytesPerPix=$6", i,
        plane_params.width[i], plane_params.height[i], plane_params.pitch[i],
        plane_params.offset[i], plane_params.psize[i], plane_params.bytesPerPix[i]);
  }
}

Status NvBufSurfaceWrapper::MapForCpu() {
  const int kAllBuffers = -1;
  const int kAllPlanes = -1;

  int retval = NvBufSurfaceMap(nvbuf_surf_, kAllBuffers, kAllPlanes, NVBUF_MAP_READ_WRITE);
  if (retval < 0) {
    return error::Internal("NvBufSurfaceMap failed.");
  }

  NvBufSurfaceSyncForCpu(nvbuf_surf_, kAllBuffers, kAllPlanes);

  return Status::OK();
}

bool NvBufSurfaceWrapper::IsMapped() const {
  bool mapped = false;
  if (nvbuf_surf_ != nullptr) {
    // Check the first plane only. Assume the rest will be in sync.
    mapped = nvbuf_surf_->surfaceList[0].mappedAddr.addr[0] != nullptr;
  }
  return mapped;
}

}  // namespace argus
}  // namespace devices
}  // namespace gem
}  // namespace gml
