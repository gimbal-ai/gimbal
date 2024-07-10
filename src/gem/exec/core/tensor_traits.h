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

#include "src/common/base/base.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/core/tensor.h"

namespace gml::gem::exec::core {

namespace internal {

template <typename T, typename = void>
struct has_reshape_fn : std::false_type {};

template <typename T>
struct has_reshape_fn<T, typename std::enable_if_t<std::is_invocable_r_v<
                             Status, decltype(&T::Reshape), T&, const TensorShape&>>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_datatype_fn : std::false_type {};

template <typename T>
struct has_set_datatype_fn<
    T,
    typename std::enable_if_t<std::is_invocable_r_v<void, decltype(&T::SetDataType), T&, DataType>>>
    : std::true_type {};

}  // namespace internal

template <typename T>
class TensorTraits {
 public:
  static constexpr bool HasReshape() { return internal::has_reshape_fn<T>::value; }
  static constexpr bool HasSetDataType() { return internal::has_set_datatype_fn<T>::value; }
};

}  // namespace gml::gem::exec::core
