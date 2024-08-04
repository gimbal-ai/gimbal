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

#include "src/common/base/array_ref.h"

#include <gtest/gtest.h>

#include "src/common/testing/testing.h"

namespace gml {

TEST(ArrayRefTest, VectorConstructor) {
  std::vector<int64_t> vec{0, 1, 2, 3};

  auto arr_ref = ArrayRef(vec);

  ASSERT_EQ(4, arr_ref.size());
  EXPECT_EQ(0, arr_ref[0]);
  EXPECT_EQ(1, arr_ref[1]);
  EXPECT_EQ(2, arr_ref[2]);
  EXPECT_EQ(3, arr_ref[3]);
}

}  // namespace gml
