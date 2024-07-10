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

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include "src/common/base/base.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/core/tensor.h"

namespace gml::gem::exec::tensorrt {

/**
 * CUDATensor represents a device-resident tensor allocated with cudaMalloc.
 *
 * For now, there is no shape or data type information associated with the buffer.
 */
class CUDATensor : public core::ReshapeableTensor {
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

}  // namespace gml::gem::exec::tensorrt
