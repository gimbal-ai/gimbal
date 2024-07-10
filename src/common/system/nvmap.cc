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
  return std::filesystem::path(sys_path) / "kernel/debug/nvmap/iovmm";
}

}  // namespace gml::system
