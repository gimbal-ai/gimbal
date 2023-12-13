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

#include "src/common/system/mac_address.h"

#include <absl/strings/str_format.h>

#include "src/common/testing/testing.h"

namespace gml::system {

using ::testing::IsEmpty;
using ::testing::Not;

TEST(MacAddrStrToInt, Basic) {
  ASSERT_OK_AND_EQ(MacAddrStrToInt("00:00:00:00:00:01"), 1);
  ASSERT_OK_AND_EQ(MacAddrStrToInt("01:02:03:04:05:06"), 0x010203040506);
  ASSERT_NOT_OK(MacAddrStrToInt("01:02:03:04:05:06:07"));
  ASSERT_NOT_OK(MacAddrStrToInt("01:02:03:04:05"));
}

TEST(MacAddrIntToStr, Basic) {
  ASSERT_EQ(MacAddrIntToStr(1), "00:00:00:00:00:01");
  ASSERT_EQ(MacAddrIntToStr(0x010203040506), "01:02:03:04:05:06");
}

class NetDeviceReaderTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_OK_AND_ASSIGN(resolver_, NetDeviceReader::Create()); }

  std::unique_ptr<NetDeviceReader> resolver_;
};

TEST_F(NetDeviceReaderTest, NonVirtualNetDevices) {
  ASSERT_OK_AND_ASSIGN(std::set<NetDevice> devices, resolver_->NonVirtualNetDevices());

  // Note this assertion expect that the test is running on a host with at least one physical
  // device. If this turns out not to be universally true for our environments, it should be
  // removed.
  ASSERT_THAT(devices, Not(IsEmpty()));

  // Print out for visual inspection.
  for (const auto& d : devices) {
    LOG(INFO) << absl::StrFormat("name=%s addr=%x", d.name(), d.mac_address().value());
  }
}

TEST_F(NetDeviceReaderTest, SystemMacAddress) {
  ASSERT_OK_AND_ASSIGN(NetDevice device, resolver_->SystemMacAddress());

  uint64_t mac_addr = device.mac_address().value();

  LOG(INFO) << absl::StrFormat("Mac Address = %x", mac_addr);

  // This is a pretty weak expectation, but since this function is host-dependent,
  // we just check that it is a 6 byte value, as MAC addresses are supposed to be.
  EXPECT_LT(mac_addr, 1ULL << 48);

  // This expectation might need to be removed if it makes the test flaky.
  // For now, just leaving this in to see if we hit this on any of our build systems.
  // If it does trigger, it warrants us investigating whether we should avoid such MAC addresses.
  EXPECT_TRUE(device.mac_address().GloballyUnique());
}

}  // namespace gml::system
