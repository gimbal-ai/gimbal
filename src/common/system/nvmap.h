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

#pragma once

#include <fstream>

#include "src/common/base/base.h"

namespace gml::system {

struct IOVMMClient {
  enum ClientType {
    IOVMM_CLIENT_TYPE_UNKNOWN = 0,
    IOVMM_CLIENT_TYPE_USER = 1,
    IOVMM_CLIENT_TYPE_TOTAL = 2,
  } client_type;

  std::string cmdline;
  int pid;
  int size_bytes;
};

Status ParseNVMapIOVMMClients(const std::filesystem::path& path, std::vector<IOVMMClient>* out);

std::filesystem::path NVMapIOVMMPath();

}  // namespace gml::system
