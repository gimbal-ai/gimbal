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

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "src/common/base/file.h"
#include "src/common/base/logging.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/fs/temp_dir.h"
#include "src/common/testing/testing.h"

namespace gml::fs {

TEST(TempDir, Basic) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TempDir> tmp_dir, TempDir::Create());
  std::filesystem::path path = tmp_dir->path();

  EXPECT_EQ(TempDirectoryPath(), path.parent_path()) << path.string();
  EXPECT_TRUE(std::filesystem::exists(path));
}

TEST(TempDir, DestructorCleansUpTempDir) {
  std::filesystem::path path;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<TempDir> tmp_dir, TempDir::Create());
    path = tmp_dir->path();
  }

  EXPECT_FALSE(std::filesystem::exists(path));
}

}  // namespace gml::fs
