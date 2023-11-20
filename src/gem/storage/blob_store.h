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

#include <utility>

#include "src/common/base/base.h"
#include "src/gem/storage/memory_blob.h"

namespace gml::gem::storage {

/**
 * BlobStore is the base class for binary blob stores.
 */
class BlobStore {
 public:
  virtual ~BlobStore() = default;

  /**
   * MapReadOnly maps a stored blob into memory for reading only. How the blob is mapped into memory
   * (mmap, copy, etc) is implementation dependent.
   */
  virtual StatusOr<std::unique_ptr<const MemoryBlob>> MapReadOnly(std::string key) const = 0;

  /**
   * FilePath returns the path to a file containing the blob.
   */
  virtual StatusOr<std::string> FilePath(std::string key) const = 0;

  /**
   * Upsert stores a new blob with the given key.
   */
  template <typename TData>
  Status Upsert(std::string key, const TData* data, size_t size) {
    static_assert(sizeof(TData) >= sizeof(char));
    constexpr size_t data_size = sizeof(TData) / sizeof(char);
    return UpsertImpl(std::move(key), reinterpret_cast<const char*>(data), size * data_size);
  }

 protected:
  virtual Status UpsertImpl(std::string key, const char* data, size_t size) = 0;
};

}  // namespace gml::gem::storage
