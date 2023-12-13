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

#include <filesystem>
#include <fstream>
#include <vector>

#include <picosha2.h>

#include "src/common/base/base.h"

namespace gml::fs {

StatusOr<std::string> GetSHA256Sum(const std::filesystem::path& file_path) {
  std::ifstream file_stream(file_path.c_str(), std::ios::binary);
  if (!file_stream.good()) {
    return error::NotFound("failed to open file");
  }
  std::vector<unsigned char> sha256_chars(picosha2::k_digest_size);
  picosha2::hash256(file_stream, sha256_chars.begin(), sha256_chars.end());
  return picosha2::bytes_to_hex_string(sha256_chars.begin(), sha256_chars.end());
}

}  // namespace gml::fs
