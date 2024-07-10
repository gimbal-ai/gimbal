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
