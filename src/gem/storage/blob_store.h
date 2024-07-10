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

#include <utility>

#include "src/common/base/base.h"

namespace gml::gem::storage {

/**
 * BlobStore is the base class for binary blob stores.
 */
class BlobStore {
 public:
  virtual ~BlobStore() = default;

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
