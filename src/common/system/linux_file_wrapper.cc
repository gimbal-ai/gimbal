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

#include "src/common/system/linux_file_wrapper.h"
#include <fcntl.h>
#include "src/common/base/error.h"
#include "src/common/base/logging.h"
#include "src/common/base/statusor.h"

namespace gml {
namespace system {

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

}  // namespace system
}  // namespace gml
