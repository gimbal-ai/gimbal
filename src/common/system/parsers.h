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
#include <string>

#include <absl/container/flat_hash_map.h>

#include "src/common/base/base.h"
#include "src/common/base/error.h"

namespace gml::system {

void ParseFromKeyValueLine(
    const std::string& line,
    const absl::flat_hash_map<std::string_view, size_t>& field_name_to_value_map,
    uint8_t* out_base);

Status ParseFromKeyValueFile(
    const std::string& fpath,
    const absl::flat_hash_map<std::string_view, size_t>& field_name_to_value_map,
    uint8_t* out_base);

template <typename T>
inline Status ReadValueFromFile(const std::string& fpath, T* arg) {
  *arg = T();
  std::ifstream f(fpath.c_str());
  if (!f.is_open()) return error::Unknown("failed to open file");
  f >> *arg;
  return (f.good() || f.eof()) ? Status::OK() : error::Unknown("failed to read file");
}

}  // namespace gml::system
