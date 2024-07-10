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
