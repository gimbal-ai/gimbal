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

#include "src/gem/storage/fs_blob_store.h"

#include <cerrno>
#include <filesystem>
#include <fstream>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/system/linux_file_wrapper.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::storage {

StatusOr<std::unique_ptr<FilesystemBlobStore>> FilesystemBlobStore::Create(
    const std::string& directory) {
  GML_RETURN_IF_ERROR(fs::CreateDirectories(directory));
  return std::unique_ptr<FilesystemBlobStore>(new FilesystemBlobStore(directory));
}

StatusOr<std::string> FilesystemBlobStore::FilePath(std::string key) const {
  auto path = directory_ / std::filesystem::path(key);
  if (!fs::Exists(path)) {
    return error::NotFound("Cannot find blob for key: $0", key);
  }
  return path.string();
}

Status FilesystemBlobStore::UpsertImpl(std::string key, const char* data, size_t size) {
  auto path = directory_ / std::filesystem::path(key);
  std::ofstream f(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!f.is_open()) {
    return error::InvalidArgument("Failed to open file for blob $0", key);
  }
  f.write(data, static_cast<int>(size));
  if (f.bad() || f.fail()) {
    return error::Internal("Failed to write to file when setting blob $0", key);
  }
  return Status::OK();
}

}  // namespace gml::gem::storage
