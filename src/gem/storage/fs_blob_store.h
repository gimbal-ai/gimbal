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
#include "src/gem/storage/memory_blob.h"

namespace gml::gem::storage {

class MemoryMappedBlob : public MemoryBlob {
 public:
  static StatusOr<std::unique_ptr<const MemoryBlob>> CreateReadOnly(
      const std::filesystem::path& path);

  ~MemoryMappedBlob() override;

 protected:
  MemoryMappedBlob(void* mmap_addr, size_t size) : mmap_addr_(mmap_addr), size_(size) {}

  const char* data() const override { return reinterpret_cast<const char*>(mmap_addr_); }
  char* data() override { return reinterpret_cast<char*>(mmap_addr_); }
  size_t size() const override { return size_; }

 private:
  void* mmap_addr_;
  size_t size_;
};

/**
 * FilesystemBlobStore stores binary blobs in the given directory on disk. It uses mmap to provide
 * InMemoryBlobs on calls to MemoryMap.
 */
class FilesystemBlobStore : public BlobStore {
 public:
  StatusOr<std::unique_ptr<const MemoryBlob>> MapReadOnly(std::string key) const override;

  static StatusOr<std::unique_ptr<FilesystemBlobStore>> Create(const std::string& directory);

 protected:
  Status UpsertImpl(std::string key, const char* data, size_t size) override;

 private:
  explicit FilesystemBlobStore(const std::string& directory) : directory_(directory) {}
  std::filesystem::path directory_;
};

}  // namespace gml::gem::storage
