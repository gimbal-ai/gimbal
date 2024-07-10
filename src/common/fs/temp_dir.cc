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

#include "src/common/fs/temp_dir.h"

#include <cstdlib>
#include <filesystem>

#include "src/common/fs/fs_wrapper.h"

namespace gml::fs {

StatusOr<std::unique_ptr<TempDir>> TempDir::Create() {
  auto tmp_path = TempDirectoryPath();
  // mkdtemp requires that the last 6 characters of the template are XXXXXX.
  auto templ_path = tmp_path / "XXXXXX";

  auto templ = templ_path.string();
  auto* path = mkdtemp(templ.data());
  if (path == nullptr) {
    return error::Internal("Failed to create temporary directory: $0", strerror(errno));
  }
  std::filesystem::path dir(path);
  return std::unique_ptr<TempDir>(new TempDir(std::move(dir)));
}

TempDir::~TempDir() { std::filesystem::remove_all(dir_); }

}  // namespace gml::fs
