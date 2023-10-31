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

#include <filesystem>
#include <iostream>

#include "src/common/base/byte_utils.h"
#include "src/common/base/utils.h"
#include "src/common/system/mac_address.h"

namespace gml {
namespace system {

// TODO(oazizi): Make this configurable. In containerized environments, we're going to want
//               to mount /sys to /host/sys and use that directory for looking up MAC addresses/
//               MAC addresses of devices inside containers are not unique.
const std::string kSysClassNet = "/sys/class/net";

StatusOr<uint64_t> MacAddrStrToInt(std::string_view mac_addr_str_in) {
  std::string_view mac_addr_str = absl::StripAsciiWhitespace(mac_addr_str_in);

  GML_ASSIGN_OR_RETURN(std::string mac_addr_bytes,
                       AsciiHexToBytes<std::string>(std::string(mac_addr_str), {':'}));

  // MAC addresses are 48 bits long.
  constexpr size_t kMacAddrBytes = 6;

  if (mac_addr_bytes.length() != kMacAddrBytes) {
    return error::Internal("Unexpected length ($0) for mac address $1", mac_addr_bytes.length(),
                           mac_addr_str);
  }

  uint64_t mac_addr = utils::BEndianBytesToInt<uint64_t, kMacAddrBytes>(mac_addr_bytes);

  return mac_addr;
}

StatusOr<std::set<NetDevice>> NonVirtualNetDevices() {
  std::set<NetDevice> net_devices;

  if (std::filesystem::exists(kSysClassNet) && std::filesystem::is_directory(kSysClassNet)) {
    for (const auto& entry : std::filesystem::directory_iterator(kSysClassNet)) {
      std::filesystem::path device_name = entry.path().filename();
      std::filesystem::path addr_file = entry / std::filesystem::path("address");

      if (!std::filesystem::exists(addr_file)) {
        continue;
      }

      auto addr_file_canonical = std::filesystem::canonical(addr_file).string();

      GML_ASSIGN_OR_RETURN(std::string mac_addr_str, ReadFileToString(addr_file));

      // Don't use the mac address of a virtual device.
      if (absl::StrContains(addr_file_canonical, "virtual")) {
        continue;
      }

      GML_ASSIGN_OR_RETURN(uint64_t mac_addr, MacAddrStrToInt(mac_addr_str));
      net_devices.emplace(device_name, mac_addr);
    }
  }

  return net_devices;
}

StatusOr<NetDevice> SystemMacAddress() {
  GML_ASSIGN_OR_RETURN(std::set<NetDevice> devices, NonVirtualNetDevices());

  // Scan for the first MAC address that is globally unique.
  for (const auto& device : devices) {
    if (!device.mac_address().GloballyUnique()) {
      continue;
    }
    return device;
  }

  // If we couldn't find any MAC address that is globally unique, then try again,
  // but this time allow locally-administered (not globally unique) MAC addresses.
  for (const auto& device : devices) {
    LOG(WARNING) << "Returning a Mac address that is locally administered (not globally unique).";
    return device;
  }

  return error::NotFound("Error: Could not find a non-virtual net device in $0.", kSysClassNet);
}

}  // namespace system
}  // namespace gml
