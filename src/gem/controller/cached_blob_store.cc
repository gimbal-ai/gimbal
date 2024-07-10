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

#include "src/gem/controller/cached_blob_store.h"

#include <filesystem>
#include <fstream>

#include "src/common/base/statusor.h"
#include "src/common/fs/fs_utils.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/gem/storage/fs_blob_store.h"

DEFINE_string(blob_store_dir, gflags::StringFromEnv("GML_BLOB_STORE_DIR", "/gml/cache/"),
              "Path to store blobs with the FilesystemBlobStore");

namespace gml::gem::controller {

StatusOr<std::unique_ptr<CachedBlobStore>> CachedBlobStore::Create(
    std::shared_ptr<FileDownloader> downloader) {
  auto blob_store = std::unique_ptr<CachedBlobStore>(
      new CachedBlobStore(FLAGS_blob_store_dir, std::move(downloader)));
  GML_RETURN_IF_ERROR(blob_store->Init());
  return blob_store;
}

CachedBlobStore::CachedBlobStore(const std::filesystem::path& directory,
                                 std::shared_ptr<FileDownloader> downloader)
    : directory_(directory),
      cas_directory_(directory / "cas"),
      by_key_directory_(directory / "by_key"),
      downloader_(std::move(downloader)) {}

Status CachedBlobStore::Init() {
  GML_RETURN_IF_ERROR(fs::CreateDirectories(directory_));
  GML_RETURN_IF_ERROR(fs::CreateDirectories(cas_directory_));
  GML_RETURN_IF_ERROR(fs::CreateDirectories(by_key_directory_));
  return Status::OK();
}

Status CachedBlobStore::EnsureBlobExists(const std::string& key, const std::string& sha256sum,
                                         size_t size) {
  {
    absl::MutexLock lock(&map_mu_);
    auto it = sha_to_filemd_.find(key);
    if (it != sha_to_filemd_.end()) {
      // Key was found we can safely return.
      return Status::OK();
    }
  }

  // Check if we already have the file and then "restore" it into the map.
  bool need_to_download = true;
  auto path = cas_directory_ / sha256sum;  //
  std::error_code ec;
  size_t fs_file_size = std::filesystem::file_size(path, ec);
  if (!ec) {
    // Check the size and sha.
    if (fs_file_size == size) {
      GML_ASSIGN_OR_RETURN(std::string file_sha, fs::GetSHA256Sum(path));
      need_to_download = (sha256sum != file_sha);
    }
  }

  if (need_to_download) {
    // Request the file and download it.
    GML_RETURN_IF_ERROR(downloader_->BlockingDownload(sole::rebuild(key), sha256sum, size, path));
  }

  // Add the file to the map.
  {
    absl::MutexLock lock(&map_mu_);
    sha_to_filemd_.emplace(key, FileMetadata(path, sha256sum, size));
  }
  return Status::OK();
}

StatusOr<std::string> CachedBlobStore::FilePath(std::string key) const {
  absl::MutexLock lock(&map_mu_);
  auto it = sha_to_filemd_.find(key);
  if (it != sha_to_filemd_.end()) {
    return it->second.path().string();
  }
  auto path = by_key_directory_ / key;
  if (!std::filesystem::exists(path)) {
    return error::NotFound("Cannot find blob for key: $0", key);
  }
  return path.string();
}

Status CachedBlobStore::UpsertImpl(std::string key, const char* data, size_t size) {
  auto path = by_key_directory_ / std::filesystem::path(key);
  std::ofstream f(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!f.is_open()) {
    return error::InvalidArgument("Failed to open file for blob $0", key);
  }
  f.write(data, static_cast<int>(size));
  if (f.bad() || f.fail()) {
    return error::Internal("Failed to write to file when setting blob $0", key);
  }
  f.close();

  GML_ASSIGN_OR_RETURN(std::string file_sha, fs::GetSHA256Sum(path));

  // Add the file to the map.
  {
    absl::MutexLock lock(&map_mu_);
    sha_to_filemd_.emplace(key, FileMetadata(path, file_sha, size));
  }
  return Status::OK();
};

}  // namespace gml::gem::controller
