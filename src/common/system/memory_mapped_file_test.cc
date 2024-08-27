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

#include "src/common/system/memory_mapped_file.h"

#include <fcntl.h>

#include <fstream>

#include "src/common/base/file.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/fs/temp_file.h"
#include "src/common/testing/testing.h"

namespace gml::system {

const std::array expected_data = {0x00, 0xc2, 0x80, 0xe0, 0xa0, 0x80, 0x0a};

TEST(MemoryMappedFile, DataIsAccessibleFromMMap) {
  auto path = bazel::RunfilePath("src/common/system/testdata/non_utf8_file.txt");

  ASSERT_OK_AND_ASSIGN(auto contents, ReadFileToString(path, std::ios_base::in));
  ASSERT_OK_AND_ASSIGN(auto mmap_file, MemoryMappedFile::MapReadOnly(path));

  ASSERT_EQ(expected_data.size(), mmap_file->size());
  for (int i = 0; i < static_cast<int>(expected_data.size()); ++i) {
    EXPECT_EQ(expected_data[i], mmap_file->data()[i]);
  }
}

}  // namespace gml::system
