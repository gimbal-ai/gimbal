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

#include "src/common/base/base.h"
#include "src/gem/build/core/execution_context_builder.h"
#include "src/gem/build/plugin/tensorrt/model_builder.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/plugins/registry.h"

namespace gml::gem::build::tensorrt {

using ::gml::gem::exec::tensorrt::ExecutionContext;

static constexpr std::string_view kPluginName = "tensorrt";

void RegisterPluginOrDie(plugins::Registry* plugin_registry) {
  plugin_registry
      ->RegisterExecContextBuilderOrDie<core::DefaultExecutionContextBuilder<ExecutionContext>>(
          kPluginName);
  plugin_registry->RegisterModelBuilderOrDie<ModelBuilder>(kPluginName);
}

GML_REGISTER_PLUGIN(RegisterPluginOrDie);

}  // namespace gml::gem::build::tensorrt
