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

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace exec {
namespace tensorrt {

/**
 * CUDATensor represents a device-resident tensor allocated with cudaMalloc.
 *
 * For now, there is no shape or data type information associated with the buffer.
 */
class CUDATensor {
 public:
  static StatusOr<std::unique_ptr<CUDATensor>> Create(size_t size) {
    void* cuda_buffer;
    cudaError_t error = cudaMalloc(&cuda_buffer, size);
    if (error != cudaSuccess) {
      return Status(gml::types::CODE_SYSTEM, "Failed to allocate CUDATensor");
    }
    return std::make_unique<CUDATensor>(cuda_buffer, size);
  }
  ~CUDATensor() {
    if (cuda_buffer_ != nullptr) {
      cudaError_t error = cudaFree(cuda_buffer_);
      ECHECK(error == cudaSuccess) << "Error freeing CUDATensor";
    }
  }

  void* data() { return cuda_buffer_; }
  size_t size() { return size_; }

  explicit CUDATensor(void* cuda_buffer, size_t size) : cuda_buffer_(cuda_buffer), size_(size) {}

 private:
  void* cuda_buffer_;
  size_t size_;
};

}  // namespace tensorrt
}  // namespace exec
}  // namespace gem
}  // namespace gml
