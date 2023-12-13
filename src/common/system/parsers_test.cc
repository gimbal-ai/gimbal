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

#include "src/common/system/parsers.h"

#include <fstream>
#include <memory>

#include "src/common/fs/temp_file.h"
#include "src/common/testing/status.h"
#include "src/common/testing/testing.h"

namespace gml::system {

TEST(ReadValueFromFile, BasicInt) {
  auto tf = fs::TempFile::Create();
  std::string tmp_file_path = tf->path();  //
  std::ofstream ofs(tmp_file_path,
                    std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs << 12345;
  ofs.close();

  int val;
  EXPECT_OK(ReadValueFromFile(tmp_file_path, &val));
  EXPECT_EQ(12345, val);
}

TEST(ReadValueFromFile, BasicDouble) {
  auto tf = fs::TempFile::Create();
  std::string tmp_file_path = tf->path();
  std::ofstream ofs(tmp_file_path,
                    std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs << 12345;
  ofs.close();

  double val;
  EXPECT_OK(ReadValueFromFile(tmp_file_path, &val));
  EXPECT_DOUBLE_EQ(12345, val);
}

TEST(ReadValueFromFile, MissingFile) {
  double val;
  EXPECT_NOT_OK(ReadValueFromFile("/should/not/exist.txt", &val));
}

}  // namespace gml::system
