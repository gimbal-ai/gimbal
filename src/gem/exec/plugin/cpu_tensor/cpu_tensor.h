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
