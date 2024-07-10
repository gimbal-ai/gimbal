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

#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/core/tensor_pool.h"

namespace gml::gem::exec::cpu_tensor {

// TODO(james): We should look into better options for our CPUTensor (eg. Eigen), but this
// is a simple way forward for now.
class CPUTensor : public core::ReshapeableTensor {
 public:
  static StatusOr<std::unique_ptr<CPUTensor>> Create(size_t size) {
    return std::make_unique<CPUTensor>(size);
  }
  explicit CPUTensor(size_t size) : data_(size) {}
  uint8_t* data() { return data_.data(); }
  const uint8_t* data() const { return data_.data(); }
  size_t size() const { return data_.size(); }

  template <core::DataType TDataType>
  typename core::DataTypeTraits<TDataType>::value_type* TypedData() {
    return reinterpret_cast<typename core::DataTypeTraits<TDataType>::value_type*>(data_.data());
  }

  template <core::DataType TDataType>
  const typename core::DataTypeTraits<TDataType>::value_type* TypedData() const {
    return reinterpret_cast<const typename core::DataTypeTraits<TDataType>::value_type*>(
        data_.data());
  }

 private:
  std::vector<uint8_t> data_;
};

using CPUTensorPool = core::TensorPool<CPUTensor>;
using CPUTensorPtr = CPUTensorPool::PoolManagedPtr;

}  // namespace gml::gem::exec::cpu_tensor
