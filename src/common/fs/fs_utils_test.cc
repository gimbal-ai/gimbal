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

#include "src/common/fs/fs_utils.h"

#include <gtest/gtest.h>

#include "src/common/bazel/runfiles.h"
#include "src/common/testing/testing.h"

std::filesystem::path kTestDataBasePath = "src/common/fs/testdata";

namespace gml::fs {

namespace {
std::string GetPathToTestDataFile(std::string_view fname) {
  return bazel::RunfilePath(kTestDataBasePath / fname);
}
}  // namespace

TEST(GetSHA256Sum, basic) {
  auto sha_or = GetSHA256Sum(GetPathToTestDataFile("testfile.txt"));
  EXPECT_OK_AND_EQ(sha_or, "649b8b471e7d7bc175eec758a7006ac693c434c8297c07db15286788c837154a");
}

TEST(GetSHA256Sum, missingFile) {
  auto sha_or = GetSHA256Sum(GetPathToTestDataFile("amissingfile.txt"));
  ASSERT_NOT_OK(sha_or);
}

}  // namespace gml::fs
