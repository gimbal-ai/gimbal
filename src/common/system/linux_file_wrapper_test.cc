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

#include <errno.h>
#include <fcntl.h>

#include "src/common/system/linux_file_wrapper.h"
#include "src/common/testing/testing.h"

namespace gml {
namespace system {

TEST(LinuxFile, file_closed_after_destructor) {
  int fd;
  {
    ASSERT_OK_AND_ASSIGN(auto file, LinuxFile::Open("/tmp", O_TMPFILE | O_RDWR, S_IRWXU));
    fd = file->fd();
  }

  // Check that the file descriptor is no longer open.
  auto ret = fcntl(fd, F_GETFL);
  ASSERT_EQ(-1, ret);
  ASSERT_EQ(EBADF, errno);
}

}  // namespace system
}  // namespace gml
