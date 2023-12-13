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
