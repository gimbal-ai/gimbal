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

#include "src/common/base/bimap.h"

#include <gtest/gtest.h>

#include "src/common/testing/testing.h"

namespace gml {

TEST(BiMapTest, InsertAndRemove) {
  BiMap<std::string, int> bimap;

  static constexpr int kCount = 1000;

  // Insert a bunch of keys and values.
  for (int i = 0; i < kCount; i++) {
    ASSERT_OK(bimap.Insert(std::to_string(i), i));
  }
  ASSERT_EQ(bimap.size(), kCount);
  ASSERT_FALSE(bimap.empty());

  // Check that we can access by key.
  for (int i = 0; i < kCount; i++) {
    ASSERT_OK_AND_ASSIGN(const int* k, bimap.KeyToValuePtr(std::to_string(i)));
    EXPECT_EQ(*k, i);
  }

  // Check that we can access by value.
  for (int i = 0; i < kCount; i++) {
    ASSERT_OK_AND_ASSIGN(const std::string* v, bimap.ValueToKeyPtr(i));
    EXPECT_EQ(*v, std::to_string(i));
  }

  // Erase half the elements by key.
  for (int i = 0; i < kCount / 2; i++) {
    bimap.EraseKey(std::to_string(i));
  }
  ASSERT_EQ(bimap.size(), kCount / 2);

  // Check that the erased elements are gone.
  for (int i = 0; i < kCount; i++) {
    if (i < kCount / 2) {
      ASSERT_NOT_OK(bimap.KeyToValuePtr(std::to_string(i)));
    } else {
      ASSERT_OK_AND_ASSIGN(const int* k, bimap.KeyToValuePtr(std::to_string(i)));
      EXPECT_EQ(*k, i);
    }
  }

  // Erase the rest of the elements by value.
  for (int i = kCount / 2; i < kCount; i++) {
    bimap.EraseValue(i);
  }
  ASSERT_EQ(bimap.size(), 0);
  ASSERT_TRUE(bimap.empty());
}

TEST(BiMapTest, InsertDuplicates) {
  BiMap<std::string, int> bimap;

  static constexpr int kCount = 1000;

  // Insert a bunch of keys and values.
  for (int i = 0; i < kCount; i++) {
    ASSERT_OK(bimap.Insert(std::to_string(i), i));
  }
  ASSERT_EQ(bimap.size(), kCount);

  for (int i = 0; i < kCount; i++) {
    ASSERT_NOT_OK(bimap.Insert(std::to_string(i), kCount + i));
    ASSERT_NOT_OK(bimap.Insert(std::to_string(1000 + i), i));
  }
  ASSERT_EQ(bimap.size(), kCount);
}

TEST(BiMapTest, Erase) {
  BiMap<std::string, int> bimap;

  ASSERT_OK(bimap.Insert("auger", 1));
  ASSERT_OK(bimap.Insert("hammer", 2));

  EXPECT_EQ(bimap.size(), 2);

  // These don't exist.
  bimap.EraseKey("screwdriver");
  bimap.EraseValue(42);

  EXPECT_EQ(bimap.size(), 2);

  bimap.EraseKey("auger");

  EXPECT_EQ(bimap.size(), 1);

  bimap.EraseValue(2);

  EXPECT_EQ(bimap.size(), 0);
}

TEST(BiMapTest, Contains) {
  BiMap<std::string, int> bimap;

  ASSERT_OK(bimap.Insert("auger", 1));
  ASSERT_OK(bimap.Insert("hammer", 2));

  EXPECT_TRUE(bimap.ContainsKey("auger"));
  EXPECT_TRUE(bimap.ContainsValue(2));

  EXPECT_FALSE(bimap.ContainsKey("screwdriver"));
  EXPECT_FALSE(bimap.ContainsValue(42));
}

}  // namespace gml
