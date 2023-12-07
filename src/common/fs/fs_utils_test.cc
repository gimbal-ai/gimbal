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

#include "fs_utils.h"

#include <gtest/gtest.h>

#include "src/common/testing/test_environment.h"
#include "src/common/testing/testing.h"

std::filesystem::path kTestDataBasePath = "src/common/fs/testdata";

namespace gml::fs {

namespace {
std::string GetPathToTestDataFile(std::string_view fname) {
  return testing::BazelRunfilePath(kTestDataBasePath / fname);
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
