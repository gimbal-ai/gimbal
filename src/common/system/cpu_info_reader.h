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

#include <memory>
#include <string>
#include <vector>

#include "src/common/base/base.h"

namespace gml::system {

struct CPUFrequencyInfo {
  int64_t cpu_num;
  int64_t cpu_freq_hz;
};

class CPUInfoReader {
 public:
  // Creates a new CPU info reader.
  static StatusOr<std::unique_ptr<CPUInfoReader>> Create();

  virtual ~CPUInfoReader() = default;

  // Init must be called before any other functions are called.
  virtual Status Init() = 0;

  // ReadCPUFrequencies appends the cpu frequencies to the passed in array.
  virtual Status ReadCPUFrequencies(std::vector<CPUFrequencyInfo>* cpu_freqs) const = 0;

 protected:
  CPUInfoReader() = default;
};

}  // namespace gml::system
