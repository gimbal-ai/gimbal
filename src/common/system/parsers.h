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
