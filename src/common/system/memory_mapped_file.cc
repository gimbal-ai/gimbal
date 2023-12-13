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

#include "src/common/system/memory_mapped_file.h"

#include "src/common/base/error.h"
#include "src/common/base/logging.h"
#include "src/common/system/linux_file_wrapper.h"

namespace gml::system {

MemoryMappedFile::~MemoryMappedFile() {
  auto ret = munmap(mmap_addr_, size_);
  if (ret == -1) {
    PLOG(ERROR) << "Failed to unmap memory mapped addr";
  }
}

StatusOr<std::unique_ptr<const MemoryMappedFile>> MemoryMappedFile::MapReadOnly(
    const std::string& path) {
  GML_ASSIGN_OR_RETURN(auto file, LinuxFile::Open(path, O_RDONLY));
  auto fsize = lseek(file->fd(), 0, SEEK_END);
  if (fsize == static_cast<off_t>(-1)) {
    return error::Internal("Failed to seek to end of file $0: $1", path, std::strerror(errno));
  }
  auto ret = lseek(file->fd(), 0, SEEK_SET);
  if (ret == static_cast<off_t>(-1)) {
    return error::Internal("Failed to seek to beginning of file $0: $1", path,
                           std::strerror(errno));
  }
  void* addr = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, file->fd(), 0);
  if (addr == MAP_FAILED) {
    return error::Internal("Failed to memory map file $0: $1", path, std::strerror(errno));
  }
  // mmap allows for the file descriptor to be closed, without invalidating the memory mapped
  // region. So we don't need to hang onto the LinuxFile.
  return std::unique_ptr<const MemoryMappedFile>(new MemoryMappedFile(addr, fsize));
}

}  // namespace gml::system
