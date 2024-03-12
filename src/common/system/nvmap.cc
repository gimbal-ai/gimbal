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

#include "src/common/system/nvmap.h"

#include <fstream>

#include "src/common/base/error.h"
#include "src/common/system/config.h"

namespace gml::system {

Status ParseNVMapIOVMMClients(const std::filesystem::path& path, std::vector<IOVMMClient>* out) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return error::Internal("failed to open iovmm clients file at $0", path.string());
  }

  /**
  CLIENT                        PROCESS      PID        SIZE
  user                              gem     4481     118372K
  user                      gnome-shell     2831       5384K
  user                             Xorg     1731      18140K
  user                   nvargus-daemon     1012          0K
  total                                              141896K
  */
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<absl::string_view> splits = absl::StrSplit(line, ' ', absl::SkipWhitespace());
    if (splits.size() < 1) {
      return error::Internal("failed to parse format of iovmm clients file");
    }
    int split_idx = 0;
    IOVMMClient client = {};
    if (splits[split_idx] == "user") {
      client.client_type = IOVMMClient::IOVMM_CLIENT_TYPE_USER;
    } else if (splits[split_idx] == "total") {
      client.client_type = IOVMMClient::IOVMM_CLIENT_TYPE_TOTAL;
    } else {
      continue;
    }

    ++split_idx;

    if (client.client_type == IOVMMClient::IOVMM_CLIENT_TYPE_USER) {
      if (splits.size() < 4) {
        return error::Internal("failed to parse format of iovmm clients file");
      }
      client.cmdline = splits[split_idx];
      ++split_idx;

      if (!absl::SimpleAtoi(splits[split_idx], &client.pid)) {
        return error::Internal("failed to parse PID from iovmm clients file");
      }
      ++split_idx;
    }

    auto size_str = absl::StripSuffix(splits[split_idx], "K");
    if (size_str == splits[split_idx]) {
      return error::Internal(
          "failed to parse size from iovmm clients file: expected size in Kilobytes got $0",
          splits[split_idx]);
    }
    if (!absl::SimpleAtoi(size_str, &client.size_bytes)) {
      return error::Internal("failed to parse size from iovmm clients file");
    }
    // The clients file uses kibibytes as the unit.
    client.size_bytes *= 1024;

    out->push_back(std::move(client));
  }

  return Status::OK();
}

std::filesystem::path NVMapIOVMMPath() {
  const auto& sys_path = Config::GetInstance().sys_path();
  return sys_path / "kernel/debug/nvmap/iovmm";
}

}  // namespace gml::system
