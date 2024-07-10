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

#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/model.h"

namespace gml::gem::build::core {

/**
 * ExecutionContextBuilder is the base class for plugin ExecutionContextBuilders.
 */
class ExecutionContextBuilder {
 public:
  using ExecutionContext = ::gml::gem::exec::core::ExecutionContext;
  using Model = ::gml::gem::exec::core::Model;

  virtual ~ExecutionContextBuilder() = default;
  virtual StatusOr<std::unique_ptr<ExecutionContext>> Build(Model* model) = 0;
};

/**
 * DefaultExecutionContextBuilder is a convenience class for ExecutionContexts that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TExecutionContext>
class DefaultExecutionContextBuilder : public ExecutionContextBuilder {
 public:
  ~DefaultExecutionContextBuilder() override = default;

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
                  "(Model*) in order to use core::DefaultExecutionContextBuilder");
    return Status();
  }
};

}  // namespace gml::gem::build::core
