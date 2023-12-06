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

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::capabilities::core {

using gml::internal::api::core::v1::DeviceCapabilities;

/**
 * CapabilityLister is the base class for plugin CapabilityListers.
 */
class CapabilityLister {
 public:
  virtual ~CapabilityLister() = default;
  virtual Status Populate(DeviceCapabilities* cap) = 0;
};

}  // namespace gml::gem::capabilities::core
