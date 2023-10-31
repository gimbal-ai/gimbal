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

#include <nvbufsurface.h>

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace devices {
namespace argus {

// A wrapper around the image buf fd, with managed resources.
class NvBufSurfaceWrapper : public NotCopyable {
 public:
  static StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> Create(NvBufSurface* nv_buf_surf);

  NvBufSurfaceWrapper(NvBufSurfaceWrapper&& other);
  ~NvBufSurfaceWrapper();

  void DumpInfo() const;

  /**
   * Maps the buffer for CPU access.
   * Information about where the different planes are located are populated in the internal
   * NvBufSurface data structure, which can be accessed via surface().mappedAddr.
   */
  Status MapForCpu();

  /**
   * Returns true of MapForCpu() has previously been called.
   */
  bool IsMapped() const;

  const NvBufSurfaceParams& surface() const { return nvbuf_surf_->surfaceList[0]; }

 private:
  NvBufSurfaceWrapper(NvBufSurface* nvbuf_surf);

  NvBufSurface* nvbuf_surf_ = nullptr;
};

}  // namespace argus
}  // namespace devices
}  // namespace gem
}  // namespace gml
