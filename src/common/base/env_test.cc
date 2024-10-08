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

#include "src/common/base/env.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace gml {

TEST(GetEnvTest, ResultsAreAsExpected) {
  const std::string rand_string = "abcdefglkljadkfjadkfj";
  EXPECT_EQ(std::nullopt, GetEnv(rand_string));

  ASSERT_EQ(0, setenv(rand_string.c_str(), "test", /*overwrite*/ 1));
  auto value_or = GetEnv(rand_string);
  ASSERT_TRUE(value_or.has_value());
  EXPECT_EQ("test", value_or.value_or(""));
}

}  // namespace gml
