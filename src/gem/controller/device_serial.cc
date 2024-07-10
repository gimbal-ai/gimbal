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

#include "src/gem/controller/device_serial.h"

namespace gml::gem::controller {
DEFINE_string(device_serial, gflags::StringFromEnv("GML_DEVICE_SERIAL", ""),
              "Force set the serial number / ID for the device. Note this needs to be unique "
              "across devices.");
DEFINE_string(sys_class_net_path,
              gflags::StringFromEnv("GML_SYS_CLASS_NET_PATH",
                                    system::NetDeviceReader::kDefaultSysClassNet.string().c_str()),
              "Set path to /sys/class/net for when the host /sys/class/net is mounted at a "
              "different path inside the container");

// The device serial string. We selectively use the passed in flag, or
// the mac_address. If a unique_id cannot be found, we will currently error out.
StatusOr<std::string> GetDeviceSerial() {
  if (!FLAGS_device_serial.empty()) {
    return FLAGS_device_serial;
  }

  GML_ASSIGN_OR_RETURN(auto dev_reader, system::NetDeviceReader::Create(FLAGS_sys_class_net_path));
  GML_ASSIGN_OR_RETURN(system::NetDevice dev, dev_reader->SystemMacAddress());
  if (!dev.mac_address().GloballyUnique()) {
    return error::FailedPrecondition("Failed to get unique mac address");
  }
  return dev.mac_address().str();
}

}  // namespace gml::gem::controller
