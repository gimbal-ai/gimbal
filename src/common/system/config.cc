/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

#include <utility>

#include "src/common/system/config.h"

#include <unistd.h>

#include "src/common/base/base.h"
#include "src/common/fs/fs_wrapper.h"

namespace gml::system {

DEFINE_string(sys_path, gflags::StringFromEnv("GML_SYS_PATH", "/sys"),
              "The path to the sys directory.");

DEFINE_string(host_path, gflags::StringFromEnv("GML_HOST_PATH", ""),
              "The path to the host root directory.");

#include <ctime>

Config::Config() : host_path_(FLAGS_host_path), sys_path_(FLAGS_sys_path) {}

int64_t Config::PageSizeBytes() const { return sysconf(_SC_PAGESIZE); }

int64_t Config::KernelTicksPerSecond() const { return sysconf(_SC_CLK_TCK); }

int64_t Config::KernelTickTimeNS() const {
  return static_cast<int64_t>(static_cast<int64_t>(1E9) / KernelTicksPerSecond());
}

const Config& Config::GetInstance() {
  static Config config;
  return config;
}

}  // namespace gml::system
