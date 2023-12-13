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

#include "src/common/system/parsers.h"

#include <fstream>
#include <vector>

namespace gml::system {

void ParseFromKeyValueLine(
    const std::string& line,
    const absl::flat_hash_map<std::string_view, size_t>& field_name_to_value_map,
    uint8_t* out_base) {
  std::vector<std::string_view> split = absl::StrSplit(line, ':', absl::SkipWhitespace());
  if (split.size() >= 2) {
    const auto& key = split[0];
    const auto& val = split[1];

    const auto& it = field_name_to_value_map.find(key);
    // Key not found in map, we can just go to next iteration of loop.
    if (it == field_name_to_value_map.end()) {
      return;
    }

    size_t offset = it->second;
    auto val_ptr = reinterpret_cast<int64_t*>(out_base + offset);

    bool ok = false;
    if (absl::EndsWith(val, " kB")) {
      // Convert kB to bytes. proc seems to only use kB as the unit if it's present
      // else there are no units.
      const std::string_view trimmed_val = absl::StripSuffix(val, " kB");
      ok = absl::SimpleAtoi(trimmed_val, val_ptr);
      *val_ptr *= 1024;
    } else {
      ok = absl::SimpleAtoi(val, val_ptr);
    }

    if (!ok) {
      *val_ptr = -1;
    }
  }
  return;
}

Status ParseFromKeyValueFile(
    const std::string& fpath,
    const absl::flat_hash_map<std::string_view, size_t>& field_name_to_value_map,
    uint8_t* out_base) {
  std::ifstream ifs;
  ifs.open(fpath);
  if (!ifs) {
    return error::Internal("Failed to open file $0.", fpath);
  }

  std::string line;
  size_t read_count = 0;
  while (std::getline(ifs, line)) {
    ParseFromKeyValueLine(line, field_name_to_value_map, out_base);

    // Check to see if we have read all the fields, if so we can skip the
    // rest. We assume no duplicates.
    if (read_count == field_name_to_value_map.size()) {
      break;
    }
  }

  return Status::OK();
}

}  // namespace gml::system
