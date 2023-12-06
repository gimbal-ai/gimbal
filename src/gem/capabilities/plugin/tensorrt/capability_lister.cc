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

#include "src/gem/capabilities/plugin/tensorrt/capability_lister.h"

namespace gml::gem::capabilities::tensorrt {

Status CapabilityLister::Populate(DeviceCapabilities* cap) {
  cap->add_model_runtimes()->set_type(
      internal::api::core::v1::
          DeviceCapabilities_ModelRuntimeInfo_ModelRuntimeType_MODEL_RUNTIME_TYPE_TENSORRT);
  return Status::OK();
}

}  // namespace gml::gem::capabilities::tensorrt
