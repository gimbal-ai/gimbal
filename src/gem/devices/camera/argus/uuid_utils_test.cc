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
