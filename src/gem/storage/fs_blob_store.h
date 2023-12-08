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

#include <filesystem>

#include "src/common/base/base.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::storage {

/**
 * FilesystemBlobStore stores binary blobs in the given directory on disk.
 */
class FilesystemBlobStore : public BlobStore {
 public:
  StatusOr<std::string> FilePath(std::string key) const override;

  static StatusOr<std::unique_ptr<FilesystemBlobStore>> Create(const std::string& directory);

 protected:
  Status UpsertImpl(std::string key, const char* data, size_t size) override;

 private:
  explicit FilesystemBlobStore(const std::string& directory) : directory_(directory) {}
  std::filesystem::path directory_;
};

}  // namespace gml::gem::storage
