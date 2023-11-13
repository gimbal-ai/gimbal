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

#include <string>

#include <absl/strings/strip.h>
#include <absl/strings/substitute.h>
#include <magic_enum.hpp>

#include "src/common/base/status.h"
#include "src/common/typespb/status.pb.h"

namespace gml::error {

// Declare convenience functions:
// error::InvalidArgument(...)
// error::IsInvalidArgument(stat)
#define DECLARE_ERROR(FUNC, CONST)                                         \
  template <typename... Args>                                              \
  Status FUNC(std::string_view format, Args... args) {                     \
    return Status(::gml::types::CONST, absl::Substitute(format, args...)); \
  }                                                                        \
  inline bool Is##FUNC(const Status& status) { return status.code() == ::gml::types::CONST; }

DECLARE_ERROR(Cancelled, CODE_CANCELLED)
DECLARE_ERROR(Unknown, CODE_UNKNOWN)
DECLARE_ERROR(InvalidArgument, CODE_INVALID_ARGUMENT)
DECLARE_ERROR(DeadlineExceeded, CODE_DEADLINE_EXCEEDED)
DECLARE_ERROR(NotFound, CODE_NOT_FOUND)
DECLARE_ERROR(AlreadyExists, CODE_ALREADY_EXISTS)
DECLARE_ERROR(PermissionDenied, CODE_PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, CODE_UNAUTHENTICATED)
DECLARE_ERROR(Internal, CODE_INTERNAL)
DECLARE_ERROR(Unimplemented, CODE_UNIMPLEMENTED)
DECLARE_ERROR(ResourceUnavailable, CODE_RESOURCE_UNAVAILABLE)
DECLARE_ERROR(System, CODE_SYSTEM)
DECLARE_ERROR(FailedPrecondition, CODE_FAILED_PRECONDITION)

#undef DECLARE_ERROR

inline char toupper(char c) {
  return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

inline char tolower(char c) {
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

inline std::string CodeToString(gml::types::Code code) {
  std::string_view code_str_view = magic_enum::enum_name(code);
  if (code_str_view.empty()) {
    return "Unknown error_code";
  }

  std::string_view stripped_code_str = absl::StripPrefix(code_str_view, "CODE_");
  std::string code_str(stripped_code_str);
  // Example transformation: INVALID_ARGUMENT -> Invalid Argument
  int last = ' ';
  std::for_each(code_str.begin(), code_str.end(), [&last](char& c) {
    if (c == '_') {
      c = ' ';
    } else {
      c = (last == ' ') ? toupper(c) : tolower(c);
    }
    last = static_cast<unsigned char>(c);
  });

  return code_str;
}

}  // namespace gml::error
