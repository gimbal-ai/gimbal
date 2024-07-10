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

#include <fcntl.h>
#include <sys/mman.h>

#include <memory>

#include "src/common/base/statusor.h"

namespace gml::system {

class MemoryMappedFile {
 public:
  ~MemoryMappedFile();

  static StatusOr<std::unique_ptr<const MemoryMappedFile>> MapReadOnly(const std::string& path);

  const uint8_t* data() const { return static_cast<uint8_t*>(mmap_addr_); }
  size_t size() const { return size_; }

 protected:
  explicit MemoryMappedFile(void* mmap_addr, size_t size) : mmap_addr_(mmap_addr), size_(size) {}

 private:
  void* mmap_addr_;
  size_t size_;
};

}  // namespace gml::system
