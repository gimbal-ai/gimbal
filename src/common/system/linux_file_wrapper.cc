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

#include "src/common/system/linux_file_wrapper.h"

#include <fcntl.h>

#include "src/common/base/error.h"
#include "src/common/base/logging.h"
#include "src/common/base/statusor.h"

namespace gml::system {

StatusOr<std::unique_ptr<LinuxFile>> LinuxFile::Open(const std::string& path, int flags, int mode) {
  int fd = open(path.c_str(), flags, mode);
  if (fd < 0) {
    return error::Internal("Failed to open file at path $0: $1", path, std::strerror(errno));
  }

  return std::unique_ptr<LinuxFile>(new LinuxFile(fd));
}

LinuxFile::~LinuxFile() {
  auto ret = close(fd_);

  ECHECK(ret == 0) << absl::Substitute("Failed to close file descriptor: $0", std::strerror(errno));
}

}  // namespace gml::system
