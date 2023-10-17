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

#include <list>
#include <memory>
#include <unordered_set>

#include <absl/base/internal/spinlock.h>

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace core {

/**
 * TensorPool manages a pool of tensors to allow reusing tensor allocations.
 * The memory won't ever be released until the TensorPool goes out of scope.
 * The assumption is that through repeated executions of a mediapipe graph the tensors allocated
 * should be the same. If this assumption doesn't apply, consumers should avoid using a TensorPool
 * and allocate the tensors directly instead.
 *
 * The TensorPool leaves allocation up to the tensor type. The TTensor must have the following
 * methods:
 *
 *  static StatusOr<std::unique_ptr<TTensor>> Create(size_t size);
 *  size_t size() const;
 *
 * The PoolManagedPtr returned by GetTensor will return the tensor to the pool when the pointer is
 * deleted.
 */
template <typename TTensor>
class TensorPool : public gml::NotCopyable {
 public:
  class Deleter {
   public:
    Deleter(TensorPool<TTensor>* pool) : pool_(pool) {}
    void operator()(TTensor* ptr) { pool_->Release(ptr); }

   private:
    TensorPool<TTensor>* pool_;
  };

  using PoolManagedPtr = std::shared_ptr<TTensor>;

  TensorPool() : deleter_(Deleter(this)) {}

  ~TensorPool() { Clear(); }

  StatusOr<PoolManagedPtr> GetTensor(size_t tensor_size) {
    absl::base_internal::SpinLockHolder lock(&lock_);
    TTensor* ptr = nullptr;
    auto& free_list = free_lists_by_size_[tensor_size];
    if (free_list.size() > 0) {
      ptr = free_list.front();
      free_list.pop_front();
    } else {
      GML_ASSIGN_OR_RETURN(auto tensor, TTensor::Create(tensor_size));
      ptr = tensor.get();
      pool_.emplace(std::move(tensor));
    }
    return PoolManagedPtr(ptr, deleter_);
  }

  void Release(TTensor* tensor) {
    if (tensor == nullptr) {
      return;
    }
    absl::base_internal::SpinLockHolder lock(&lock_);
    auto& free_list = free_lists_by_size_[tensor->size()];
    free_list.push_back(tensor);
  }

  void Clear() {
    absl::base_internal::SpinLockHolder lock(&lock_);
    size_t num_free = 0;
    for (auto& pair : free_lists_by_size_) {
      num_free += pair.second.size();
    }
    CHECK(num_free == pool_.size())
        << "Attempted to clear TensorPool while Tensors are still in use";
    free_lists_by_size_.clear();
    pool_.clear();
  }

 private:
  Deleter deleter_;

  absl::base_internal::SpinLock lock_;
  std::unordered_map<size_t, std::list<TTensor*>> free_lists_by_size_;
  std::unordered_set<std::unique_ptr<TTensor>> pool_;
};

}  // namespace core
}  // namespace gem
}  // namespace gml
