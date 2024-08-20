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

#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/model_exec.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/utils.h"
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/build/core/execution_context_builder.h"
#include "src/gem/build/core/model_builder.h"
#include "src/gem/capabilities/core/capability_lister_builder.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/metrics/core/scraper_builder.h"
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

  std::vector<std::string> ListRegistered() {
    auto keys = TransformContainer<std::vector<std::string>>(
        builders_, [](const auto& entry) { return entry.first; });

    return keys;
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
    capability_lister_builders_.RegisterOrDie<T>(name);
  }

  template <typename T>
  void RegisterMetricsScraperOrDie(std::string_view name) {
    metrics_scraper_builders_.RegisterOrDie<T>(name);
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
    return capability_lister_builders_.Build(name);
  }

  StatusOr<std::unique_ptr<metrics::core::Scraper>> BuildMetricsScraper(
      std::string_view name, gml::metrics::MetricsSystem* metrics_system,
      gml::event::Dispatcher* dispatcher) {
    return metrics_scraper_builders_.Build(name, metrics_system, dispatcher);
  }

  std::vector<std::string> RegisteredCapabilityListers() {
    return capability_lister_builders_.ListRegistered();
  }

  std::vector<std::string> RegisteredMetricsScrapers() {
    return metrics_scraper_builders_.ListRegistered();
  }

 private:
  BuilderRegistry<build::core::ExecutionContextBuilder, exec::core::ExecutionContext,
                  exec::core::Model*>
      exec_ctx_builders_;
  BuilderRegistry<build::core::ModelBuilder, exec::core::Model, storage::BlobStore*,
                  const ::gml::internal::api::core::v1::ModelSpec&>
      model_builders_;
  BuilderRegistry<capabilities::core::CapabilityListerBuilder, capabilities::core::CapabilityLister>
      capability_lister_builders_;

  BuilderRegistry<metrics::core::ScraperBuilder, metrics::core::Scraper,
                  gml::metrics::MetricsSystem*, gml::event::Dispatcher*>
      metrics_scraper_builders_;
};

#define GML_REGISTER_PLUGIN(reg_func)                        \
  bool __register_plugin__() {                               \
    reg_func(&::gml::gem::plugins::Registry::GetInstance()); \
    return true;                                             \
  }                                                          \
  __attribute__((unused)) static bool ___plugin___ = __register_plugin__()

}  // namespace gml::gem::plugins
