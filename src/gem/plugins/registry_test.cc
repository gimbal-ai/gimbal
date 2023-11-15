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
#include "src/api/corepb/v1/model_exec.pb.h"
#include "src/gem/build/core/execution_context_builder.h"
#include "src/gem/build/core/model_builder.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/model.h"

#include "src/common/testing/testing.h"

namespace gml::gem::plugins {

using ::gml::gem::build::core::ExecutionContextBuilder;
using ::gml::gem::build::core::ModelBuilder;
using ::gml::gem::exec::core::ExecutionContext;
using ::gml::gem::exec::core::Model;
using ::gml::internal::api::core::v1::ModelSpec;

class SimpleExecutionContext : public ExecutionContext {
 public:
  explicit SimpleExecutionContext(size_t i) : i_(i) {}

  size_t i() { return i_; }

 private:
  size_t i_;
};

class SimpleExecutionContextBuilder : public ExecutionContextBuilder {
 public:
  StatusOr<std::unique_ptr<ExecutionContext>> Build(Model*) override {
    return std::unique_ptr<ExecutionContext>(new SimpleExecutionContext(10));
  }
};

class SimpleModel : public Model {
 public:
  explicit SimpleModel(size_t i) : i_(i) {}
  size_t i() { return i_; }

 private:
  size_t i_;
};

TEST(Registry, SimpleRegisterExecCtx) {
  auto& plugin_registry = Registry::GetInstance();

  plugin_registry.RegisterExecContextBuilderOrDie<SimpleExecutionContextBuilder>("simple");

  SimpleModel model(1);
  auto exec_ctx_or_s = plugin_registry.BuildExecutionContext("simple", &model);
  ASSERT_OK(exec_ctx_or_s);
  auto exec_ctx = static_cast<SimpleExecutionContext*>(exec_ctx_or_s.ValueOrDie().get());
  EXPECT_EQ(10, exec_ctx->i());
}

class SimpleModelBuilder : public ModelBuilder {
 public:
  StatusOr<std::unique_ptr<Model>> Build(storage::BlobStore*, const ModelSpec&) override {
    return std::unique_ptr<Model>(new SimpleModel(15));
  }
};

TEST(Registry, SimpleRegisterModel) {
  auto& plugin_registry = Registry::GetInstance();

  plugin_registry.RegisterModelBuilderOrDie<SimpleModelBuilder>("simple");

  ModelSpec spec;
  auto model_or_s = plugin_registry.BuildModel("simple", nullptr, spec);
  ASSERT_OK(model_or_s);
  auto model = static_cast<SimpleModel*>(model_or_s.ValueOrDie().get());
  EXPECT_EQ(15, model->i());
}

}  // namespace gml::gem::plugins
