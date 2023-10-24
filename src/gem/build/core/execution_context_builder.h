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
#include "src/gem/exec/core/model.h"

namespace gml {
namespace gem {
namespace build {
namespace core {

using ::gml::gem::exec::core::ExecutionContext;
using ::gml::gem::exec::core::Model;

/**
 * ExecutionContextBuilder is the base class for plugin ExecutionContextBuilders.
 */
class ExecutionContextBuilder {
 public:
  virtual ~ExecutionContextBuilder() {}
  virtual StatusOr<std::unique_ptr<ExecutionContext>> Build(Model* model) = 0;
};

/**
 * DefaultExecutionContextBuilder is a convenience class for ExecutionContexts that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TExecutionContext>
class DefaultExecutionContextBuilder : public ExecutionContextBuilder {
 public:
  ~DefaultExecutionContextBuilder() override {}

  StatusOr<std::unique_ptr<ExecutionContext>> Build(Model* model) override {
    return BuildInternal(model);
  }

 private:
  template <typename Q = TExecutionContext, std::enable_if_t<std::is_constructible_v<Q>>* = nullptr>
  StatusOr<std::unique_ptr<ExecutionContext>> BuildInternal(Model*) {
    return std::unique_ptr<ExecutionContext>(new Q);
  }

  template <typename Q = TExecutionContext,
            std::enable_if_t<std::is_constructible_v<Q, Model*>>* = nullptr>
  StatusOr<std::unique_ptr<ExecutionContext>> BuildInternal(Model* model) {
    return std::unique_ptr<ExecutionContext>(new Q(model));
  }

  template <typename Q = TExecutionContext,
            std::enable_if_t<!std::is_constructible_v<Q, Model*> && !std::is_constructible_v<Q>>* =
                nullptr>
  StatusOr<std::unique_ptr<ExecutionContext>> BuildInternal(Model*) {
    static_assert(sizeof(Q) == 0,
                  "ExecutionContext must have default constructor or constructor with args "
                  "(core::Model*) in order to use core::DefaultExecutionContextBuilder");
    return Status();
  }
};

}  // namespace core
}  // namespace build
}  // namespace gem
}  // namespace gml
