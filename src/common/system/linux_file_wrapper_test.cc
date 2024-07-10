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

#include <cerrno>

#include "src/common/testing/testing.h"

namespace gml::system {

TEST(LinuxFile, FileClosedAfterDestructor) {
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

}  // namespace gml::system
