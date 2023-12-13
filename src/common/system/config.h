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

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "src/common/base/base.h"

namespace gml::system {

DECLARE_string(host_path);

/**
 * This interface provides access to global system config.
 */
class Config : public NotCopyable {
 public:
  /**
   * Create an OS specific SystemConfig instance.
   * @return const reference to SystemConfig.
   */
  static const Config& GetInstance();

  /**
   * Resets the underlying static instance. Used for testing purposes.
   */
  static void ResetInstance();

  /**
   * Get the page size in the kernel.
   * @return page size in bytes.
   */
  int64_t PageSizeBytes() const;

  /**
   * Get the Kernel ticks per second.
   * @return int kernel ticks per second.
   */
  int64_t KernelTicksPerSecond() const;

  /**
   * Get the Kernel tick time in nanoseconds.
   * @return int kernel ticks time.
   */
  int64_t KernelTickTimeNS() const;

  /**
   * If recording `nsecs` from bpf, this function can be used to
   * convert the result into realtime.
   */
  uint64_t ConvertToRealTime(uint64_t monotonic_time) const;

  /**
   * Converts a path to host relative path, for when this binary is running inside a container.
   */
  std::filesystem::path ToHostPath(const std::filesystem::path& p) const;

  const std::filesystem::path& host_path() const { return host_path_; }
  const std::filesystem::path& sys_path() const { return sys_path_; }

 private:
  explicit Config();

  const std::filesystem::path host_path_;
  const std::filesystem::path sys_path_;
};

}  // namespace gml::system
