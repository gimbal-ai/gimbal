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

#include "src/gem/plugins/registry.h"
#include "src/gem/core/build/execution_context_builder.h"
#include "src/gem/core/exec/context.h"

#include "src/common/testing/testing.h"

namespace gml {
namespace gem {
namespace plugins {

class SimpleExecutionContext : public core::ExecutionContext {
 public:
  explicit SimpleExecutionContext(size_t i) : i_(i) {}

  size_t i() { return i_; }

 private:
  size_t i_;
};

class SimpleExecutionContextBuilder : public core::ExecutionContextBuilder {
 public:
  StatusOr<std::unique_ptr<core::ExecutionContext>> Build(const core::spec::ExecutionSpec&) {
    return std::unique_ptr<core::ExecutionContext>(new SimpleExecutionContext(10));
  }
};

TEST(Registry, simple_register) {
  Registry plugin_registry;

  plugin_registry.RegisterExecContextBuilderOrDie<SimpleExecutionContextBuilder>("simple");

  core::spec::ExecutionSpec spec;
  auto exec_ctx_or_s = plugin_registry.BuildExecutionContext("simple", spec);
  ASSERT_OK(exec_ctx_or_s);
  auto exec_ctx = static_cast<SimpleExecutionContext*>(exec_ctx_or_s.ValueOrDie().get());
  EXPECT_EQ(10, exec_ctx->i());
}

}  // namespace plugins
}  // namespace gem
}  // namespace gml
