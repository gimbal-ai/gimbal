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

#include <string>

#include <sole.hpp>

namespace gml::gem::controller {

struct GEMInfo {
  GEMInfo() = default;
  // Identification information for the gem.
  sole::uuid id;
  uint32_t pid = 0;
  std::string hostname;
  // The IP address;
  std::string ip_address;
};

}  // namespace gml::gem::controller
