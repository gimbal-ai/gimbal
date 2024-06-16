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

#include <vector>

#include <magic_enum.hpp>

#include "src/common/base/base.h"

namespace gml::gem::exec::core {

enum class DataType {
  UNKNOWN = 0,
  FLOAT32 = 1,
  INT32 = 2,
  INT64 = 3,
};

template <DataType TDataType>
class DataTypeTraits {
 public:
  using value_type = uint8_t;
};

template <>
class DataTypeTraits<DataType::FLOAT32> {
 public:
  using value_type = float;
};

template <>
class DataTypeTraits<DataType::INT32> {
 public:
  using value_type = int32_t;
};

template <>
class DataTypeTraits<DataType::INT64> {
 public:
  using value_type = int64_t;
};

static inline size_t DataTypeByteSize(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return sizeof(DataTypeTraits<DataType::FLOAT32>::value_type);
    case DataType::INT32:
      return sizeof(DataTypeTraits<DataType::INT32>::value_type);
    case DataType::INT64:
      return sizeof(DataTypeTraits<DataType::INT64>::value_type);
    default:
      return sizeof(DataTypeTraits<DataType::UNKNOWN>::value_type);
  }
}

}  // namespace gml::gem::exec::core
