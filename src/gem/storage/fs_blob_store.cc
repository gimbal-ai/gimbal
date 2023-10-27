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

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <filesystem>
#include <fstream>

#include "src/common/base/base.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/system/linux_file_wrapper.h"
#include "src/gem/storage/blob_store.h"
#include "src/gem/storage/fs_blob_store.h"
#include "src/gem/storage/memory_blob.h"

namespace gml {
namespace gem {
namespace storage {

MemoryMappedBlob::~MemoryMappedBlob() {
  auto ret = munmap(mmap_addr_, size_);
  if (ret == -1) {
    PLOG(ERROR) << "Failed to unmap memory mapped addr";
  }
}

StatusOr<std::unique_ptr<const MemoryBlob>> MemoryMappedBlob::CreateReadOnly(
    std::filesystem::path path) {
  GML_ASSIGN_OR_RETURN(auto file, system::LinuxFile::Open(path.string(), O_RDONLY));
  auto fsize = lseek(file->fd(), 0, SEEK_END);
  if (fsize == static_cast<off_t>(-1)) {
    return error::Internal("Failed to seek to end of file $0: $1", path.string(),
                           std::strerror(errno));
  }
  auto ret = lseek(file->fd(), 0, SEEK_SET);
  if (ret == static_cast<off_t>(-1)) {
    return error::Internal("Failed to seek to beginning of file $0: $1", path.string(),
                           std::strerror(errno));
  }
  void* addr = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, file->fd(), 0);
  if (addr == MAP_FAILED) {
    return error::Internal("Failed to memory map file $0: $1", path.string(), std::strerror(errno));
  }
  // mmap allows for the file descriptor to be closed, without invalidating the memory mapped
  // region. So we don't need to hang onto the LinuxFile.
  return std::unique_ptr<const MemoryBlob>(new MemoryMappedBlob(addr, fsize));
}

StatusOr<std::unique_ptr<FilesystemBlobStore>> FilesystemBlobStore::Create(std::string directory) {
  GML_RETURN_IF_ERROR(fs::CreateDirectories(directory));
  return std::unique_ptr<FilesystemBlobStore>(new FilesystemBlobStore(directory));
}

StatusOr<std::unique_ptr<const MemoryBlob>> FilesystemBlobStore::MapReadOnly(
    std::string key) const {
  auto path = directory_ / std::filesystem::path(key);

  if (!fs::Exists(path)) {
    return std::unique_ptr<const MemoryBlob>(nullptr);
  }

  return MemoryMappedBlob::CreateReadOnly(path);
}

Status FilesystemBlobStore::UpsertImpl(std::string key, const char* data, size_t size) {
  auto path = directory_ / std::filesystem::path(key);
  std::ofstream f(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!f.is_open()) {
    return error::InvalidArgument("Failed to open file for blob $0", key);
  }
  f.write(data, size);
  if (f.bad() || f.fail()) {
    return error::Internal("Failed to write to file when setting blob $0", key);
  }
  return Status::OK();
}

}  // namespace storage
}  // namespace gem
}  // namespace gml