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

#include "src/common/system/memory_mapped_file.h"

#include <fcntl.h>

#include <fstream>

#include "src/common/base/file.h"
#include "src/common/fs/temp_file.h"
#include "src/common/testing/test_environment.h"
#include "src/common/testing/testing.h"

namespace gml::system {

const std::array expected_data = {0x00, 0xc2, 0x80, 0xe0, 0xa0, 0x80, 0x0a};

TEST(MemoryMappedFile, DataIsAccessibleFromMMap) {
  auto path = testing::BazelRunfilePath("src/common/system/testdata/non_utf8_file.txt");

  ASSERT_OK_AND_ASSIGN(auto contents, ReadFileToString(path, O_RDONLY));
  ASSERT_OK_AND_ASSIGN(auto mmap_file, MemoryMappedFile::MapReadOnly(path));

  ASSERT_EQ(expected_data.size(), mmap_file->size());
  for (int i = 0; i < static_cast<int>(expected_data.size()); ++i) {
    EXPECT_EQ(expected_data[i], mmap_file->data()[i]);
  }
}

}  // namespace gml::system
