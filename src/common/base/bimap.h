/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

#include <absl/container/node_hash_map.h>

#include "src/common/base/error.h"
#include "src/common/base/status.h"
#include "src/common/base/statusor.h"

namespace gml {

template <typename T1, typename T2>
class BiMap {
 public:
  BiMap() = default;

  Status Insert(const T1& key, const T2& value) {
    if (ContainsKey(key)) {
      return error::AlreadyExists("Key already exists");
    }
    if (ContainsValue(value)) {
      return error::AlreadyExists("Value already exists");
    }

    auto ret1 = key_to_value_.emplace(std::move(key), nullptr);
    auto& it1 = ret1.first;
    auto inserted1 = ret1.second;
    DCHECK(inserted1);

    auto ret2 = value_to_key_.emplace(std::move(value), nullptr);
    auto& it2 = ret2.first;
    auto inserted2 = ret2.second;
    DCHECK(inserted2);

    // Each map's value is a pointer to the other map's key.
    it1->second = &(it2->first);
    it2->second = &(it1->first);

    return Status::OK();
  }

  bool ContainsKey(const T1& key) const { return key_to_value_.find(key) != key_to_value_.end(); }

  bool ContainsValue(const T2& value) const {
    return value_to_key_.find(value) != value_to_key_.end();
  }

  StatusOr<const T2*> KeyToValuePtr(const T1& key) const {
    auto it = key_to_value_.find(key);
    if (it != key_to_value_.end()) {
      return it->second;
    }
    return error::NotFound("Key not found");
  }

  StatusOr<const T1*> ValueToKeyPtr(const T2& value) const {
    auto it = value_to_key_.find(value);
    if (it != value_to_key_.end()) {
      return it->second;
    }
    return error::NotFound("Value not found");
  }

  StatusOr<T2> KeyToValue(const T1& key) const {
    GML_ASSIGN_OR_RETURN(auto ptr, KeyToValuePtr(key));
    DCHECK(ptr != nullptr);
    return *ptr;
  }

  StatusOr<T1> ValueToKey(const T2& value) const {
    GML_ASSIGN_OR_RETURN(auto ptr, ValueToKeyPtr(value));
    DCHECK(ptr != nullptr);
    return *ptr;
  }

  void EraseKey(const T1& key) {
    auto it = key_to_value_.find(key);
    if (it != key_to_value_.end()) {
      value_to_key_.erase(*(it->second));
      key_to_value_.erase(it);
    }
  }

  void EraseValue(const T2& value) {
    auto it = value_to_key_.find(value);
    if (it != value_to_key_.end()) {
      key_to_value_.erase(*(it->second));
      value_to_key_.erase(it);
    }
  }

  size_t size() const { return key_to_value_.size(); }

  bool empty() const { return key_to_value_.empty(); }

  void clear() {
    key_to_value_.clear();
    value_to_key_.clear();
  }

 private:
  // Use node_hash_map instead of flat_hash_map to ensure pointer stability.
  absl::node_hash_map<T1, const T2*> key_to_value_;
  absl::node_hash_map<T2, const T1*> value_to_key_;
};

}  // namespace gml
