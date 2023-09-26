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

#include <cassert>
#include <string>
#include <utility>

#include "src/common/base/macros.h"
#include "src/common/base/status.h"
#include "src/common/base/statuspb/status.pb.h"

namespace gml {

// Concept and some code borrowed
// from : tensorflow/stream_executor/lib/statusor.h
template <typename T>
class StatusOr {
  template <typename U>
  friend class StatusOr;

 public:
  // Construct a new StatusOr with Status::UNKNOWN status
  StatusOr()
      : status_(gml::statuspb::UNKNOWN,
                "Default constructed StatusOr should not be used, "
                "did you mistakenly return {}?") {}

  StatusOr(const Status& status);  // NOLINT

  StatusOr(const T& value);  // NOLINT

  // Conversion copy constructor, T must be copy constructable from U
  template <typename U>
  explicit StatusOr(const StatusOr<U>& other) : status_(other.status_), value_(other.value_) {}

  // Rvalue-reference overloads of the other constructors and assignment
  // operators, to support move-only types and avoid unnecessary copying.
  StatusOr(T&& value);  // NOLINT

  // Move conversion operator to avoid unnecessary copy.
  // T must be assignable from U.
  // Not marked with explicit so the implicit conversion can happen.
  template <typename U>
  StatusOr(StatusOr<U>&& other)  // NOLINT
      : status_(std::move(other.status_)) {
    if (status_.ok()) {
      value_ = std::move(other.value_);
    }
  }

  // Conversion assignment operator, T must be assignable from U
  template <typename U>
  StatusOr& operator=(const StatusOr<U>& other) {
    status_ = other.status_;
    value_ = other.value_;
    return *this;
  }

  // Move assignment operator to avoid unnecessary copy.
  // T must be assignable from U
  template <typename U>
  StatusOr& operator=(StatusOr<U>&& other) {
    status_ = std::move(other.status_);
    value_ = std::move(other.value_);
    return *this;
  }

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK.
  const Status& status() const { return status_; }

  // Returns this->status().ok()
  bool ok() const { return status_.ok(); }

  // Returns this->status().code()
  gml::statuspb::Code code() const { return status_.code(); }

  // Returns this->status().msg()
  std::string msg() const { return status_.msg(); }

  const T& ValueOrDie() const;
  T& ValueOrDie();

  // Moves the current value.
  T ConsumeValueOrDie();

  // ValueOr()/ConsumeValueOr() return a pre-specified value if the StatusOr is not OK.

  // Note this variant performs a copy. It does not return a reference like ValueOrDie().
  // Returning a reference doesn't work because the argument may be an rvalue.
  T ValueOr(const T& val);

  // This variants are more efficient because they do not copy.
  T ConsumeValueOr(T&& val);

  std::string ToString() const { return status().ToString(); }

  template <typename U>
  struct IsNull {
    // For non-pointer U, a reference can never be NULL.
    static inline bool IsValueNull(const U& /* t */) { return false; }
  };

  template <typename U>
  struct IsNull<U*> {
    static inline bool IsValueNull(const U* t) { return t == NULL; }
  };

 private:
  Status status_;
  T value_;
};

template <typename T>
StatusOr<T>::StatusOr(const T& value) : value_(value) {}

template <typename T>
const T& StatusOr<T>::ValueOrDie() const {
  GML_CHECK_OK(status_);
  return value_;
}

template <typename T>
T& StatusOr<T>::ValueOrDie() {
  GML_CHECK_OK(status_);
  return value_;
}

template <typename T>
T StatusOr<T>::ConsumeValueOrDie() {
  GML_CHECK_OK(status_);
  return std::move(value_);
}

template <typename T>
T StatusOr<T>::ValueOr(const T& val) {
  return status_.ok() ? value_ : val;
}

template <typename T>
T StatusOr<T>::ConsumeValueOr(T&& val) {
  return status_.ok() ? std::move(value_) : std::move(val);
}

template <typename T>
StatusOr<T>::StatusOr(const Status& status) : status_(status) {
  DCHECK(!status_.ok()) << "Should not pass OK status to constructor";
  if (status.ok()) {
    status_ = Status(gml::statuspb::INTERNAL,
                     "Status::OK is not a valid constructor argument to StatusOr<T>");
  }
}

template <typename T>
StatusOr<T>::StatusOr(T&& value) {
  if constexpr (!std::is_pointer<T>::value) {
    value_ = std::move(value);
  } else {
    value_ = value;
  }
}

// GML_UNUSED(__s__) is useful if the caller does not use the status.
#define GML_ASSIGN_OR_IMPL(statusor, lhs, rexpr, ...) \
  auto statusor = (rexpr);                            \
  if (!statusor.ok()) {                               \
    auto& __s__ = statusor;                           \
    GML_UNUSED(__s__);                                \
    __VA_ARGS__;                                      \
  }                                                   \
  lhs = std::move(statusor.ValueOrDie())

// When using GML_ASSIGN_OR(), use  '__s__' to access the statusor object in the 'or' case.
// See GML_ASSIGN_OR_RETURN for an example.
#define GML_ASSIGN_OR(lhs, rexpr, ...) \
  GML_ASSIGN_OR_IMPL(GML_CONCAT_NAME(__status_or_value__, __COUNTER__), lhs, rexpr, __VA_ARGS__)

#define GML_ASSIGN_OR_RETURN(lhs, rexpr) GML_ASSIGN_OR(lhs, rexpr, return __s__.status())

// Be careful using this, since it will exit the whole binary.
// Meant for use in top-level main() of binaries.
#define GML_ASSIGN_OR_EXIT(lhs, rexpr) \
  GML_ASSIGN_OR(lhs, rexpr, LOG(ERROR) << __s__.msg(); exit(1);)

// Adapter for status.
template <typename T>
inline Status StatusAdapter(const StatusOr<T>& s) noexcept {
  return s.status();
}

// This enables GMock to print readable description of the tested value.
template <typename T>
std::ostream& operator<<(std::ostream& os, const StatusOr<T>& status_or) {
  os << status_or.ToString();
  if (status_or.ok()) {
    os << " and holds: " << status_or.ValueOrDie();
  }
  return os;
}

}  // namespace gml
