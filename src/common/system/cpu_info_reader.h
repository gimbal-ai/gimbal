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

#include <absl/container/flat_hash_map.h>

#include "src/common/base/base.h"

namespace gml::system {

struct CPUFrequencyInfo {
  int64_t cpu_num;
  int64_t cpu_freq_hz;
};

class CPUInfoReader {
 public:
  CPUInfoReader() = default;
  Status Init();
  Status ReadCPUFrequencies(std::vector<CPUFrequencyInfo>* cpu_freqs) const;
};

}  // namespace gml::system
