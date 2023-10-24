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
#include "src/gem/core/build/model_builder.h"
#include "src/gem/core/exec/context.h"
#include "src/gem/core/exec/model.h"
#include "src/gem/core/spec/model.pb.h"

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
  StatusOr<std::unique_ptr<core::ExecutionContext>> Build(core::Model*) {
    return std::unique_ptr<core::ExecutionContext>(new SimpleExecutionContext(10));
  }
};

class SimpleModel : public core::Model {
 public:
  explicit SimpleModel(size_t i) : i_(i) {}
  size_t i() { return i_; }

 private:
  size_t i_;
};

TEST(Registry, simple_register_exec_ctx) {
  Registry plugin_registry;

  plugin_registry.RegisterExecContextBuilderOrDie<SimpleExecutionContextBuilder>("simple");

  SimpleModel model(1);
  auto exec_ctx_or_s = plugin_registry.BuildExecutionContext("simple", &model);
  ASSERT_OK(exec_ctx_or_s);
  auto exec_ctx = static_cast<SimpleExecutionContext*>(exec_ctx_or_s.ValueOrDie().get());
  EXPECT_EQ(10, exec_ctx->i());
}

class SimpleModelBuilder : public core::ModelBuilder {
 public:
  StatusOr<std::unique_ptr<core::Model>> Build(const core::spec::ModelSpec&) {
    return std::unique_ptr<core::Model>(new SimpleModel(15));
  }
};

TEST(Registry, simple_register_model) {
  Registry plugin_registry;

  plugin_registry.RegisterModelBuilderOrDie<SimpleModelBuilder>("simple");

  core::spec::ModelSpec spec;
  auto model_or_s = plugin_registry.BuildModel("simple", spec);
  ASSERT_OK(model_or_s);
  auto model = static_cast<SimpleModel*>(model_or_s.ValueOrDie().get());
  EXPECT_EQ(15, model->i());
}

}  // namespace plugins
}  // namespace gem
}  // namespace gml
