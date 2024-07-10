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

#include "src/gem/devices/camera/argus/uuid_utils.h"

#include "src/common/testing/testing.h"

namespace gml {

TEST(ToProto, UUIDBasic) {
  auto uuid = sole::rebuild("ea8aa095-697f-49f1-b127-d50e5b6e2645");
  Argus::UUID argus_uuid = gem::devices::argus::ToArgusUUID(uuid);

  EXPECT_EQ(0xea8aa095, argus_uuid.time_low);
  EXPECT_EQ(0x697f, argus_uuid.time_mid);
  EXPECT_EQ(0x49f1, argus_uuid.time_hi_and_version);

  EXPECT_EQ(0xb127, argus_uuid.clock_seq);

  EXPECT_EQ(0xd5, argus_uuid.node[0]);
  EXPECT_EQ(0x0e, argus_uuid.node[1]);
  EXPECT_EQ(0x5b, argus_uuid.node[2]);
  EXPECT_EQ(0x6e, argus_uuid.node[3]);
  EXPECT_EQ(0x26, argus_uuid.node[4]);
  EXPECT_EQ(0x45, argus_uuid.node[5]);
}

TEST(UUIDUtils, RegressionTest) {
  for (int i = 0; i < 1000; i++) {
    auto uuid = sole::uuid4();
    Argus::UUID argus_uuid = gem::devices::argus::ToArgusUUID(uuid);
    auto res = gem::devices::argus::ParseUUID(argus_uuid);
    EXPECT_EQ(res.str(), uuid.str());
  }
}

}  // namespace gml
