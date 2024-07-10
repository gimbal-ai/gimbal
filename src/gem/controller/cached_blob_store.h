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

#include <filesystem>

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>

#include "src/common/event/event.h"
#include "src/gem/controller/file_downloader.h"
#include "src/gem/controller/grpc_bridge.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::controller {

class FileMetadata {
 public:
  FileMetadata() = default;
  FileMetadata(std::filesystem::path path, std::string sha256sum, size_t size)
      : path_(std::move(path)), sha256sum_(std::move(sha256sum)), size_(size) {}

  const std::filesystem::path& path() const { return path_; }
  const std::string& sha256sum() const { return sha256sum_; }
  size_t size() const { return size_; }

 private:
  std::filesystem::path path_;
  std::string sha256sum_;
  size_t size_;
};

class CachedBlobStore : public storage::BlobStore {
 public:
  CachedBlobStore() = delete;
  static StatusOr<std::unique_ptr<CachedBlobStore>> Create(
      std::shared_ptr<FileDownloader> downloader);

  Status Init();
  /**
   * EnsureBlobExists make sure the blob exists in the store. Otherwise, it will
   * block until the file is downloaded.
   */
  Status EnsureBlobExists(const std::string& key, const std::string& sha256sum, size_t size);

  StatusOr<std::string> FilePath(std::string key) const override;

 protected:
  Status UpsertImpl(std::string key, const char* data, size_t size) override;

 private:
  CachedBlobStore(const std::filesystem::path& directory,
                  std::shared_ptr<FileDownloader> downloader);

  std::filesystem::path directory_;
  std::filesystem::path cas_directory_;
  std::filesystem::path by_key_directory_;

  std::shared_ptr<FileDownloader> downloader_;

  // Mutexe guarding the metadata maps.
  mutable absl::Mutex map_mu_;
  // Map from key -> file metadata.
  absl::flat_hash_map<std::string, FileMetadata> sha_to_filemd_ ABSL_GUARDED_BY(map_mu_);
};

}  // namespace gml::gem::controller
