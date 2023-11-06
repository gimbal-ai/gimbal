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

#include "src/common/base/base.h"
#include "src/gem/build/core/execution_context_builder.h"
#include "src/gem/build/plugin/tensorrt/model_builder.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/plugins/registry.h"

namespace gml {
namespace gem {
namespace build {
namespace tensorrt {

using ::gml::gem::exec::tensorrt::ExecutionContext;

static constexpr std::string_view kPluginName = "tensorrt";

void RegisterPluginOrDie(plugins::Registry* plugin_registry) {
  plugin_registry
      ->RegisterExecContextBuilderOrDie<core::DefaultExecutionContextBuilder<ExecutionContext>>(
          kPluginName);
  plugin_registry->RegisterModelBuilderOrDie<ModelBuilder>(kPluginName);
}

GML_REGISTER_PLUGIN(RegisterPluginOrDie);

}  // namespace tensorrt
}  // namespace build
}  // namespace gem
}  // namespace gml
