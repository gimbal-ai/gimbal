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
