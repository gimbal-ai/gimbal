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
#include "src/gem/core/build/execution_context_builder.h"
#include "src/gem/core/exec/context.h"
#include "src/gem/core/spec/execution_spec.pb.h"
#include "src/gem/plugins/tensorrt/exec/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace tensorrt {

class ExecutionContextBuilder : public core::ExecutionContextBuilder {
 public:
  StatusOr<std::unique_ptr<core::ExecutionContext>> Build(
      const core::spec::ExecutionSpec& spec) override;
};

}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
