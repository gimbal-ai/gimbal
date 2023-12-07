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
#include "src/gem/capabilities/plugin/opencv_cam/capability_lister.h"
#include "src/gem/plugins/registry.h"

namespace gml::gem::capabilities::opencv_cam {

static constexpr std::string_view kPluginName = "opencv_cam";

void RegisterPluginOrDie(plugins::Registry* plugin_registry) {
  plugin_registry
      ->RegisterCapabilityListerOrDie<core::DefaultCapabilityListerBuilder<CapabilityLister>>(
          kPluginName);
}

GML_REGISTER_PLUGIN(RegisterPluginOrDie);

}  // namespace gml::gem::capabilities::opencv_cam