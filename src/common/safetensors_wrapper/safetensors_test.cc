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

#include "src/common/safetensors_wrapper/safetensors.h"

#include "gmock/gmock.h"

#include "src/common/bazel/runfiles.h"
#include "src/common/testing/testing.h"

namespace gml::safetensors {

class SafeTensorsFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = bazel::RunfilePath("src/common/safetensors_wrapper/testdata/test.safetensors");
  }
  std::filesystem::path path_;
};

TEST_F(SafeTensorsFileTest, OpenTest) {
  // The safetensors file was generated with the following python code:
  //
  // from safetensors.torch import save_file
  // import torch
  //
  // tensors = {
  //    "b": torch.arange(12).reshape((3, 1, 4)).to(torch.uint8),
  //    "a": 0.5 * torch.ones((2, 3, 4), dtype=torch.float32),
  // }
  //
  // save_file(tensors, "./test.safetensors")

  ASSERT_OK_AND_ASSIGN(auto file, SafeTensorsFile::Open(path_));

  ASSERT_EQ(2, file->Size());
  EXPECT_THAT(file->TensorNames(), ::testing::UnorderedElementsAre("b", "a"));

  ASSERT_OK_AND_ASSIGN(auto a, file->Tensor("a"));
  EXPECT_EQ(a->DataType(), rust::Dtype::F32);
  EXPECT_THAT(a->Shape(), ::testing::ElementsAre(2, 3, 4));
  EXPECT_EQ(reinterpret_cast<const float*>(a->Data().data())[0], 0.5);

  ASSERT_OK_AND_ASSIGN(auto b, file->Tensor("b"));
  EXPECT_EQ(b->DataType(), rust::Dtype::U8);
  EXPECT_THAT(b->Shape(), ::testing::ElementsAre(3, 1, 4));

  size_t num_elems = 1;
  for (auto size : b->Shape()) {
    num_elems *= size;
  }

  std::vector<uint8_t> b_data;
  for (size_t i = 0; i < num_elems; ++i) {
    b_data.push_back(reinterpret_cast<const uint8_t*>(b->Data().data())[i]);
  }
  EXPECT_THAT(b_data, ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
}

}  // namespace gml::safetensors
