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

#include "src/common/fs/temp_dir.h"

#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "src/common/base/file.h"
#include "src/common/base/logging.h"
#include "src/common/fs/fs_wrapper.h"
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
