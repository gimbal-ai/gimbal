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

#include <vector>

#include <magic_enum.hpp>

#include "src/common/base/base.h"

namespace gml::gem::exec::core {

enum class DataType {
  UNKNOWN = 0,
  FLOAT32 = 1,
  INT32 = 2,
  INT64 = 3,
  INT8 = 4,
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

template <>
class DataTypeTraits<DataType::INT8> {
 public:
  using value_type = int8_t;
};

static inline size_t DataTypeByteSize(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return sizeof(DataTypeTraits<DataType::FLOAT32>::value_type);
    case DataType::INT32:
      return sizeof(DataTypeTraits<DataType::INT32>::value_type);
    case DataType::INT64:
      return sizeof(DataTypeTraits<DataType::INT64>::value_type);
    case DataType::INT8:
      return sizeof(DataTypeTraits<DataType::INT8>::value_type);
    default:
      return sizeof(DataTypeTraits<DataType::UNKNOWN>::value_type);
  }
}

}  // namespace gml::gem::exec::core
