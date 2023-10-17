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

#include "src/gem/core/exec/tensor_pool.h"
#include "src/common/testing/testing.h"

namespace gml {
namespace gem {
namespace core {
namespace exec {

static int gNumTensorsCreated = 0;

class TestTensor {
 public:
  explicit TestTensor(size_t size) : size_(size), cb_([]() {}) {}
  ~TestTensor() { cb_(); }
  static StatusOr<std::unique_ptr<TestTensor>> Create(size_t size) {
    gNumTensorsCreated++;
    return std::make_unique<TestTensor>(size);
  }

  size_t size() { return size_; }

  using Callback = std::function<void(void)>;
  void SetDestructorCallback(Callback func) { cb_ = func; }

 private:
  size_t size_;
  Callback cb_;
};

TEST(TensorPool, get_tensor) {
  TensorPool<TestTensor> pool;
  ASSERT_OK_AND_ASSIGN(auto tensor, pool.GetTensor(10));
  EXPECT_EQ(10, tensor->size());
}

TEST(TensorPool, ensure_not_destroyed) {
  bool destructed = false;
  {
    TensorPool<TestTensor> pool;
    ASSERT_OK_AND_ASSIGN(auto tensor, pool.GetTensor(10));

    tensor->SetDestructorCallback([&]() { destructed = true; });
    tensor.reset();

    ASSERT_FALSE(destructed) << "Tensor was destructed when returned to the pool";
  }
  ASSERT_TRUE(destructed) << "Tensor was not destructed when pool went out of scope";
}

TEST(TensorPool, ensure_reused) {
  auto initial_tensors_created = gNumTensorsCreated;
  TensorPool<TestTensor> pool;
  ASSERT_OK_AND_ASSIGN(auto tensor, pool.GetTensor(10));
  // release the tensor and allocate another one of the same size.
  tensor.reset();
  ASSERT_OK_AND_ASSIGN(tensor, pool.GetTensor(10));
  // it shouldn't have called Create again.
  ASSERT_EQ(1, gNumTensorsCreated - initial_tensors_created);
  // creating another tensor of a different size should call create again.
  ASSERT_OK_AND_ASSIGN(tensor, pool.GetTensor(20));
  ASSERT_EQ(2, gNumTensorsCreated - initial_tensors_created);
}

}  // namespace exec
}  // namespace core
}  // namespace gem
}  // namespace gml
