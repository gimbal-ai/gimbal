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

#include "src/common/base/base.h"
#include "src/gem/exec/core/tensor_pool.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor.h"

namespace gml::gem::exec::tensorrt {

using CUDATensorPool = core::TensorPool<CUDATensor>;

using CUDATensorPtr = CUDATensorPool::PoolManagedPtr;

}  // namespace gml::gem::exec::tensorrt
