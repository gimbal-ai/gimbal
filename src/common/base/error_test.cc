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

#include <absl/strings/match.h>
#include <gtest/gtest.h>
#include <iostream>

#include "src/common/base/error.h"
#include "src/common/typespb/status.pb.h"

namespace gml::error {

TEST(CodeToString, strings) {
  EXPECT_EQ("Ok", CodeToString(gml::types::CODE_OK));
  EXPECT_EQ("Cancelled", CodeToString(gml::types::CODE_CANCELLED));
  EXPECT_EQ("Unknown", CodeToString(gml::types::CODE_UNKNOWN));
  EXPECT_EQ("Invalid Argument", CodeToString(gml::types::CODE_INVALID_ARGUMENT));
  EXPECT_EQ("Deadline Exceeded", CodeToString(gml::types::CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ("Not Found", CodeToString(gml::types::CODE_NOT_FOUND));
  EXPECT_EQ("Already Exists", CodeToString(gml::types::CODE_ALREADY_EXISTS));
  EXPECT_EQ("Permission Denied", CodeToString(gml::types::CODE_PERMISSION_DENIED));
  EXPECT_EQ("Unauthenticated", CodeToString(gml::types::CODE_UNAUTHENTICATED));
  EXPECT_EQ("Internal", CodeToString(gml::types::CODE_INTERNAL));
  EXPECT_EQ("Unimplemented", CodeToString(gml::types::CODE_UNIMPLEMENTED));
}

TEST(CodeToString, UnknownError) {
  EXPECT_TRUE(
      absl::StartsWith(CodeToString(static_cast<gml::types::Code>(1024)), "Unknown error_code"));
}

TEST(ErrorDeclerations, Cancelled) {
  Status cancelled = Cancelled("test_message $0", 0);
  ASSERT_TRUE(IsCancelled(cancelled));
  ASSERT_EQ("test_message 0", cancelled.msg());
}

}  // namespace gml::error
