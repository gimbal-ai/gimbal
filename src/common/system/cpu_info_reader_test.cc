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

#include "src/common/system/cpu_info_reader.h"

#include <memory>

#include "src/common/bazel/runfiles.h"
#include "src/common/testing/status.h"
#include "src/common/testing/testing.h"

namespace gml::system {

DECLARE_string(sys_path);

constexpr char kTestDataBasePath[] = "src/common/system";

namespace {
std::string GetPathToTestDataFile(std::string_view fname) {
  return bazel::RunfilePath(std::filesystem::path(kTestDataBasePath) / fname);
}
}  // namespace

TEST(CPUInfoReader, Basic) {
  GML_SET_FOR_SCOPE(FLAGS_sys_path, GetPathToTestDataFile("testdata/sys"));
  StatusOr s = CPUInfoReader::Create();
  ASSERT_OK(s);
  auto& reader = s.ValueOrDie();

  ASSERT_OK(reader->Init());

  std::vector<CPUFrequencyInfo> cpu_freqs;
  ASSERT_OK(reader->ReadCPUFrequencies(&cpu_freqs));
  EXPECT_EQ(2, cpu_freqs.size());
  EXPECT_EQ(0, cpu_freqs[0].cpu_num);
  EXPECT_EQ(1800000000, cpu_freqs[0].cpu_freq_hz);

  EXPECT_EQ(1, cpu_freqs[1].cpu_num);
  EXPECT_EQ(1930000000, cpu_freqs[1].cpu_freq_hz);
}

}  // namespace gml::system
