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

#include "src/common/system/hostname.h"
#include "src/common/base/base.h"

namespace gml::system {
constexpr size_t kMaxHostnameSize = 128;
StatusOr<std::string> GetHostname() {
  char hostname[kMaxHostnameSize];
  int err = gethostname(hostname, sizeof(hostname));
  if (err != 0) {
    return error::Unknown("Failed to get hostname");
  }
  return std::string(hostname);
}
}  // namespace gml::system
