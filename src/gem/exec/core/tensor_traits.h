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
