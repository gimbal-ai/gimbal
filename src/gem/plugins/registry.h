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

#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/model_exec.pb.h"
#include "src/common/base/base.h"
#include "src/gem/build/core/execution_context_builder.h"
#include "src/gem/build/core/model_builder.h"
#include "src/gem/capabilities/core/capability_lister_builder.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::plugins {

/**
 * BuilderRegistry is a template for creating a registry for an arbitrary Builder type.
 *
 * Currently, plugins only register ExecutionContextBuilders, but in the future we may want to
 * register other builders.
 */
template <typename TBuilderBase, typename TBuilt, typename... Args>
class BuilderRegistry {
 public:
  template <typename TBuilder>
  void RegisterOrDie(std::string_view name) {
    auto builder = std::make_unique<TBuilder>();
    builders_.emplace(name, std::move(builder));
  }

  StatusOr<std::unique_ptr<TBuilt>> Build(std::string_view name, Args... args) {
    if (!builders_.contains(name)) {
      return Status(gml::types::CODE_INVALID_ARGUMENT,
                    absl::Substitute("'$0' Builder not registered", name));
    }
    auto* builder = builders_[name].get();
    return builder->Build(std::forward<Args>(args)...);
  }

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<TBuilderBase>> builders_;
};

/**
 * plugins::Registry stores each plugins registered objects.
 * Currently, the only registered objects are ExecutionContextBuilders.
 */
class Registry {
 public:
  static Registry& GetInstance();

  template <typename T>
  void RegisterExecContextBuilderOrDie(std::string_view name) {
    exec_ctx_builders_.RegisterOrDie<T>(name);
  }

  template <typename T>
  void RegisterModelBuilderOrDie(std::string_view name) {
    model_builders_.RegisterOrDie<T>(name);
  }

  template <typename T>
  void RegisterCapabilityListerOrDie(std::string_view name) {
    capability_listers_.RegisterOrDie<T>(name);
  }

  StatusOr<std::unique_ptr<exec::core::ExecutionContext>> BuildExecutionContext(
      std::string_view name, exec::core::Model* model) {
    return exec_ctx_builders_.Build(name, model);
  }

  StatusOr<std::unique_ptr<exec::core::Model>> BuildModel(
      std::string_view name, storage::BlobStore* store,
      const ::gml::internal::api::core::v1::ModelSpec& spec) {
    return model_builders_.Build(name, store, spec);
  }

  StatusOr<std::unique_ptr<capabilities::core::CapabilityLister>> BuildCapabilityLister(
      std::string_view name) {
    return capability_listers_.Build(name);
  }

 private:
  BuilderRegistry<build::core::ExecutionContextBuilder, exec::core::ExecutionContext,
                  exec::core::Model*>
      exec_ctx_builders_;
  BuilderRegistry<build::core::ModelBuilder, exec::core::Model, storage::BlobStore*,
                  const ::gml::internal::api::core::v1::ModelSpec&>
      model_builders_;
  BuilderRegistry<capabilities::core::CapabilityListerBuilder, capabilities::core::CapabilityLister>
      capability_listers_;
};

#define GML_REGISTER_PLUGIN(reg_func)                        \
  bool __register_plugin__() {                               \
    reg_func(&::gml::gem::plugins::Registry::GetInstance()); \
    return true;                                             \
  }                                                          \
  __attribute__((unused)) static bool ___plugin___ = __register_plugin__()

}  // namespace gml::gem::plugins
