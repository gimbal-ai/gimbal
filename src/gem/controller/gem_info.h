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

#include <string>

#include <sole.hpp>

namespace gml::gem::controller {

struct GEMInfo {
  GEMInfo() = default;
  // Identification information for the gem.
  sole::uuid id;
  uint32_t pid = 0;
  std::string hostname;
  // The IP address;
  std::string ip_address;
};

}  // namespace gml::gem::controller
