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
