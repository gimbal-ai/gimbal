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
