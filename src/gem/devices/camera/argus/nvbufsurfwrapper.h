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

#include <NvBufSurface.h>

#include "src/common/base/base.h"

namespace gml::gem::devices::argus {

// A wrapper around the image buf fd, with managed resources.
class NvBufSurfaceWrapper : public NotCopyable {
 public:
  static StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> Create(NvBufSurface* nv_buf_surf);

  NvBufSurfaceWrapper(NvBufSurfaceWrapper&& other) noexcept;
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
  explicit NvBufSurfaceWrapper(NvBufSurface* nvbuf_surf);

  NvBufSurface* nvbuf_surf_ = nullptr;
};

}  // namespace gml::gem::devices::argus
