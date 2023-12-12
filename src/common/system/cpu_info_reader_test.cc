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

#include <memory>

#include "src/common/system/cpu_info_reader.h"
#include "src/common/testing/status.h"
#include "src/common/testing/testing.h"

namespace gml::system {

TEST(CPUInfoReader, Basic) {
  auto reader = CPUInfoReader::Create();
  ASSERT_OK(reader->Init());

  std::vector<CPUFrequencyInfo> cpu_freqs;
  ASSERT_OK(reader->ReadCPUFrequencies(&cpu_freqs));

  // TODO(zasgar): Implement this.
  EXPECT_EQ(0, cpu_freqs.size());
}

}  // namespace gml::system
