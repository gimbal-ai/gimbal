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
#include "src/gem/core/exec/context.h"
#include "src/gem/core/spec/execution_spec.pb.h"

namespace gml {
namespace gem {
namespace core {

/**
 * ExecutionContextBuilder is the base class for plugin ExecutionContextBuilders.
 */
class ExecutionContextBuilder {
 public:
  virtual ~ExecutionContextBuilder() {}
  virtual StatusOr<std::unique_ptr<ExecutionContext>> Build(const spec::ExecutionSpec& spec) = 0;
};

/**
 * DefaultExecutionContextBuilder is a convenience class for ExecutionContexts that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TExecutionContext>
class DefaultExecutionContextBuilder : public ExecutionContextBuilder {
 public:
  ~DefaultExecutionContextBuilder() override {}
  StatusOr<std::unique_ptr<ExecutionContext>> Build(const spec::ExecutionSpec&) override {
    return std::unique_ptr<ExecutionContext>(new TExecutionContext);
  }
};

}  // namespace core
}  // namespace gem
}  // namespace gml
