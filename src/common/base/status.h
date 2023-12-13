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
#include <memory>
#include <string>
#include <utility>

#include <absl/status/status.h>
#include <google/protobuf/any.h>
#include <google/protobuf/message.h>

#include "src/common/base/logging.h"
#include "src/common/base/macros.h"
#include "src/common/typespb/status.pb.h"

namespace gml {

class GML_MUST_USE_RESULT Status {
 public:
  // Success status.
  Status() = default;
  Status(const Status& s) noexcept;
  Status(gml::types::Code code, const std::string& msg);
  Status(gml::types::Code code, const std::string& msg,
         std::unique_ptr<google::protobuf::Message> ctx);
  // NOLINTNEXTLINE to make it easier to return status.
  Status(const gml::types::Status& status_pb);

  void operator=(const Status& s) noexcept;

  /// Returns true if the status indicates success.
  bool ok() const { return (state_ == nullptr); }

  // Return self, this makes it compatible with StatusOr<>.
  const Status& status() const { return *this; }

  gml::types::Code code() const { return ok() ? gml::types::CODE_OK : state_->code; }

  const std::string& msg() const { return ok() ? empty_string() : state_->msg; }

  google::protobuf::Any* context() const { return ok() ? nullptr : state_->context.get(); }
  bool has_context() const { return ok() ? false : state_->context != nullptr; }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  std::string ToString() const;

  static Status OK() { return {}; }

  gml::types::Status ToProto() const;
  void ToProto(gml::types::Status* status_pb) const;

 private:
  struct State {
    // Needed for a call in status.cc.
    State() = default;
    State(const State& state) noexcept;
    State(gml::types::Code code, std::string msg, std::unique_ptr<google::protobuf::Any> context)
        : code(code), msg(std::move(msg)), context(std::move(context)) {}
    State(gml::types::Code code, std::string msg,
          std::unique_ptr<google::protobuf::Message> generic_pb_context)
        : code(code), msg(std::move(msg)) {
      if (generic_pb_context == nullptr) {
        return;
      }
      context = std::make_unique<google::protobuf::Any>();
      context->PackFrom(*generic_pb_context);
    }
    gml::types::Code code;
    std::string msg;
    std::unique_ptr<google::protobuf::Any> context;
  };

  static const std::string& empty_string() {
    static auto* empty = new std::string;
    return *empty;
  }

  // Will be null if status is OK.
  std::unique_ptr<State> state_;
};

inline Status::State::State(const State& state) noexcept : code(state.code), msg(state.msg) {
  if (!state.context) {
    context = nullptr;
    return;
  }
  context = std::unique_ptr<google::protobuf::Any>(state.context->New());
  context->CopyFrom(*state.context);
}

inline Status::Status(const Status& s) noexcept
    : state_((s.state_ == nullptr) ? nullptr : new(std::nothrow) State(*s.state_)) {}

inline void Status::operator=(const Status& s) noexcept {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    if (s.state_ == nullptr) {
      state_ = nullptr;
    } else {
      state_ = std::make_unique<State>(*s.state_);
    }
  }
}

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

template <typename T>
inline Status StatusAdapter(const T&) noexcept {
  static_assert(sizeof(T) == 0, "Implement custom status adapter, or include correct .h file.");
  return {types::CODE_UNIMPLEMENTED, "Should never get here"};
}

template <>
inline Status StatusAdapter<Status>(const Status& s) noexcept {
  return s;
}

// Conversion of proto status message.
template <>
inline Status StatusAdapter<gml::types::Status>(const gml::types::Status& s) noexcept {
  return {s};
};

inline gml::types::Code AbslCodeToStatusCode(absl::StatusCode code) noexcept {
  switch (code) {
    case absl::StatusCode::kOk:
      return gml::types::CODE_OK;
    case absl::StatusCode::kCancelled:
      return gml::types::CODE_CANCELLED;
    case absl::StatusCode::kUnknown:
      return gml::types::CODE_UNKNOWN;
    case absl::StatusCode::kInvalidArgument:
      return gml::types::CODE_INVALID_ARGUMENT;
    case absl::StatusCode::kDeadlineExceeded:
      return gml::types::CODE_DEADLINE_EXCEEDED;
    case absl::StatusCode::kNotFound:
      return gml::types::CODE_NOT_FOUND;
    case absl::StatusCode::kAlreadyExists:
      return gml::types::CODE_ALREADY_EXISTS;
    case absl::StatusCode::kPermissionDenied:
      return gml::types::CODE_PERMISSION_DENIED;
    case absl::StatusCode::kResourceExhausted:
      return gml::types::CODE_RESOURCE_UNAVAILABLE;
    case absl::StatusCode::kFailedPrecondition:
      return gml::types::CODE_FAILED_PRECONDITION;
    case absl::StatusCode::kAborted:
      return gml::types::CODE_CANCELLED;
    case absl::StatusCode::kOutOfRange:
      return gml::types::CODE_FAILED_PRECONDITION;
    case absl::StatusCode::kUnimplemented:
      return gml::types::CODE_UNIMPLEMENTED;
    case absl::StatusCode::kInternal:
      return gml::types::CODE_INTERNAL;
    case absl::StatusCode::kUnavailable:
      return gml::types::CODE_RESOURCE_UNAVAILABLE;
    case absl::StatusCode::kDataLoss:
      // We might want a more specific code here.
      return gml::types::CODE_INTERNAL;
    case absl::StatusCode::kUnauthenticated:
      return gml::types::CODE_UNAUTHENTICATED;
    default:
      return gml::types::CODE_UNKNOWN;
  }
}

inline absl::StatusCode StatusCodeToAbslCode(gml::types::Code code) noexcept {
  switch (code) {
    case gml::types::CODE_OK:
      return absl::StatusCode::kOk;
    case gml::types::CODE_CANCELLED:
      return absl::StatusCode::kCancelled;
    case gml::types::CODE_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case gml::types::CODE_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case gml::types::CODE_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case gml::types::CODE_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case gml::types::CODE_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case gml::types::CODE_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case gml::types::CODE_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
    case gml::types::CODE_INTERNAL:
      return absl::StatusCode::kInternal;
    case gml::types::CODE_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case gml::types::CODE_RESOURCE_UNAVAILABLE:
      return absl::StatusCode::kResourceExhausted;
    case gml::types::CODE_SYSTEM:
      return absl::StatusCode::kInternal;
    case gml::types::CODE_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    default:
      return absl::StatusCode::kUnknown;
  }
}

// Conversion of absl::Status.
template <>
inline Status StatusAdapter<absl::Status>(const absl::Status& s) noexcept {
  if (s.code() == absl::StatusCode::kOk) {
    return {};
  }
  return {AbslCodeToStatusCode(s.code()), std::string(s.message())};
}

template <typename T>
inline absl::Status AbslStatusAdapter(const T&) noexcept {
  static_assert(sizeof(T) == 0, "Implement custom status adapter, or include correct .h file.");
  return {absl::StatusCode::kUnimplemented, "Should never get here"};
}

template <>
inline absl::Status AbslStatusAdapter(const absl::Status& s) noexcept {
  return s;
}

template <>
inline absl::Status AbslStatusAdapter(const Status& s) noexcept {
  return {StatusCodeToAbslCode(s.code()), s.msg()};
}

}  // namespace gml

#define GML_RETURN_IF_ERROR_IMPL(__status_name__, __status) \
  do {                                                      \
    const auto&(__status_name__) = (__status);              \
    if (!(__status_name__).ok()) {                          \
      return StatusAdapter(__status_name__);                \
    }                                                       \
  } while (false)

// Early-returns the status if it is in error; otherwise, proceeds.
// The argument expression is guaranteed to be evaluated exactly once.
#define GML_RETURN_IF_ERROR(__status) \
  GML_RETURN_IF_ERROR_IMPL(GML_UNIQUE_NAME(__status__), __status)

#define GML_ABSL_RETURN_IF_ERROR_IMPL(__status_name__, __status) \
  do {                                                           \
    const auto&(__status_name__) = (__status);                   \
    if (!(__status_name__).ok()) {                               \
      return AbslStatusAdapter(__status_name__);                 \
    }                                                            \
  } while (false)

// Early-returns the status if it is in error; otherwise, proceeds.
// The argument expression is guaranteed to be evaluated exactly once.
#define GML_ABSL_RETURN_IF_ERROR(__status) \
  GML_ABSL_RETURN_IF_ERROR_IMPL(GML_UNIQUE_NAME(__status__), __status)

#define GML_EXIT_IF_ERROR(__status)   \
  {                                   \
    if (!(__status).ok()) {           \
      LOG(ERROR) << (__status).msg(); \
      exit(1);                        \
    }                                 \
  }

#define GML_CHECK_OK_PREPEND(to_call, msg)            \
  do {                                                \
    auto _s = (to_call);                              \
    CHECK(_s.ok()) << (msg) << ": " << _s.ToString(); \
  } while (false)

#ifdef NDEBUG
#define GML_DCHECK_OK(val) GML_UNUSED(val);
#else
#define GML_DCHECK_OK(val) GML_CHECK_OK_PREPEND(val, "Bad Status");
#endif

#define GML_CHECK_OK(val) GML_CHECK_OK_PREPEND(val, "Bad Status")
