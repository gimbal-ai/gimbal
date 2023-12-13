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
