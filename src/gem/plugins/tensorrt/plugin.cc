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

#include "src/gem/plugins/tensorrt/plugin.h"

#include "src/common/base/base.h"
#include "src/gem/plugins/registry.h"
#include "src/gem/plugins/tensorrt/build/execution_context_builder.h"

namespace gml {
namespace gem {
namespace tensorrt {

void RegisterPluginOrDie(plugins::Registry* plugin_registry) {
  plugin_registry->RegisterExecContextBuilderOrDie<ExecutionContextBuilder>("TensorRT");
}

}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
