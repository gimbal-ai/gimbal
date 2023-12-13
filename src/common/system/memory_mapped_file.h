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
