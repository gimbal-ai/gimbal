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
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

namespace gml {
namespace gem {
namespace exec {
namespace cpu_tensor {

class ExecutionContext : public core::ExecutionContext {
 public:
  ~ExecutionContext() override = default;

  ExecutionContext() = default;

  CPUTensorPool* TensorPool() { return &tensor_pool_; }

 private:
  CPUTensorPool tensor_pool_;
};

}  // namespace cpu_tensor
}  // namespace exec
}  // namespace gem
}  // namespace gml
