/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

#pragma once

#include <cstdint>
#include "src/common/base/base.h"

namespace gml {
namespace system {

struct MacAddress {
  uint64_t addr;

  static constexpr uint64_t kUnicastBitMask = (1ULL << 40);
  static constexpr uint64_t kGloballyUniqueBitMask = (1ULL << 41);

  bool Unicast() const { return (addr & kUnicastBitMask) == 0; }
  bool GloballyUnique() const { return (addr & kGloballyUniqueBitMask) == 0; }
};

/**
 * A NetDevice is a networking device found under /sys/class/net. It has a name and a MAC address.
 */
class NetDevice {
 public:
  NetDevice() = default;
  NetDevice(std::string name, uint64_t mac_addr)
      : name_(std::move(name)), mac_addr_(MacAddress{mac_addr}) {}

  const std::string& name() const { return name_; }
  MacAddress mac_address() const { return mac_addr_; }

  // Sort lexicographically by device name.
  bool operator<(const NetDevice& other) const { return name_ < other.name_; }

 private:
  std::string name_;
  MacAddress mac_addr_{0};
};

/**
 * Converts a MAC address in string form "00:1A:2B:3C:4D:5E", which is the form found in
 * /sys/class/net/<device>/address, to its integer representation.
 * The 00 byte is the MSB, while 5E is the LSB.
 */
StatusOr<uint64_t> MacAddrStrToInt(std::string_view mac_addr_str_in);

/**
 * Returns the set of non-virtual networking devices from the host.
 * Any virtual devices (like lo, docker and br devices) are excluded.
 * Virtual devices are those found under /sys/devices/virtual.
 */
StatusOr<std::set<NetDevice>> NonVirtualNetDevices();

/**
 * Returns the lexicographically first non-virtual device from the system that has a globally unique
 * MAC address, as determined by the corresponding MAC address bit. If no globally unique MAC
 * address bits are available, then it returns the lexicographically first physical MAC address,
 * even if it is locally administered.
 */
StatusOr<NetDevice> SystemMacAddress();

}  // namespace system
}  // namespace gml
