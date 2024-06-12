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

#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>

#include "src/common/base/status.h"

namespace gml {

constexpr gml::types::Code GRPCCodeToGMLCode(grpc::StatusCode code) noexcept {
  switch (code) {
    case grpc::StatusCode::OK:
      return gml::types::CODE_OK;
    case grpc::StatusCode::CANCELLED:
      return gml::types::CODE_CANCELLED;
    case grpc::StatusCode::UNKNOWN:
      return gml::types::CODE_UNKNOWN;
    case grpc::StatusCode::INVALID_ARGUMENT:
      return gml::types::CODE_INVALID_ARGUMENT;
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return gml::types::CODE_DEADLINE_EXCEEDED;
    case grpc::StatusCode::NOT_FOUND:
      return gml::types::CODE_NOT_FOUND;
    case grpc::StatusCode::ALREADY_EXISTS:
      return gml::types::CODE_ALREADY_EXISTS;
    case grpc::StatusCode::PERMISSION_DENIED:
      return gml::types::CODE_PERMISSION_DENIED;
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return gml::types::CODE_RESOURCE_UNAVAILABLE;
    case grpc::StatusCode::FAILED_PRECONDITION:
      return gml::types::CODE_FAILED_PRECONDITION;
    case grpc::StatusCode::ABORTED:
      return gml::types::CODE_CANCELLED;
    case grpc::StatusCode::OUT_OF_RANGE:
      return gml::types::CODE_FAILED_PRECONDITION;
    case grpc::StatusCode::UNIMPLEMENTED:
      return gml::types::CODE_UNIMPLEMENTED;
    case grpc::StatusCode::INTERNAL:
      return gml::types::CODE_INTERNAL;
    case grpc::StatusCode::UNAVAILABLE:
      return gml::types::CODE_RESOURCE_UNAVAILABLE;
    case grpc::StatusCode::DATA_LOSS:
      // We might want a more specific code here.
      return gml::types::CODE_INTERNAL;
    case grpc::StatusCode::UNAUTHENTICATED:
      return gml::types::CODE_UNAUTHENTICATED;
    default:
      return gml::types::CODE_UNKNOWN;
  }
}

constexpr grpc::StatusCode GMLCodeToGRPCCode(gml::types::Code code) noexcept {
  switch (code) {
    case gml::types::CODE_OK:
      return grpc::StatusCode::OK;
    case gml::types::CODE_CANCELLED:
      return grpc::StatusCode::CANCELLED;
    case gml::types::CODE_UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case gml::types::CODE_INVALID_ARGUMENT:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case gml::types::CODE_DEADLINE_EXCEEDED:
      return grpc::StatusCode::DEADLINE_EXCEEDED;
    case gml::types::CODE_NOT_FOUND:
      return grpc::StatusCode::NOT_FOUND;
    case gml::types::CODE_ALREADY_EXISTS:
      return grpc::StatusCode::ALREADY_EXISTS;
    case gml::types::CODE_PERMISSION_DENIED:
      return grpc::StatusCode::PERMISSION_DENIED;
    case gml::types::CODE_RESOURCE_UNAVAILABLE:
      return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case gml::types::CODE_FAILED_PRECONDITION:
      return grpc::StatusCode::FAILED_PRECONDITION;
    case gml::types::CODE_UNIMPLEMENTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    case gml::types::CODE_INTERNAL:
      return grpc::StatusCode::INTERNAL;
    case gml::types::CODE_UNAUTHENTICATED:
      return grpc::StatusCode::UNAUTHENTICATED;
    default:
      return grpc::StatusCode::UNKNOWN;
  }
}

// Conversion of grpc::Status message.
template <>
inline Status StatusAdapter<grpc::Status>(const grpc::Status& s) noexcept {
  if (s.error_code() == grpc::StatusCode::OK) {
    return {};
  }
  return {GRPCCodeToGMLCode(s.error_code()), s.error_message()};
};

template <typename T>
inline grpc::Status GRPCStatusAdapter(const T&) noexcept {
  static_assert(sizeof(T) == 0, "Implement custom status adapter, or include correct .h file.");
  return {grpc::StatusCode::UNIMPLEMENTED, "Should never get here"};
}

template <>
inline grpc::Status GRPCStatusAdapter(const gml::Status& s) noexcept {
  if (s.ok()) {
    return grpc::Status::OK;
  }
  return {GMLCodeToGRPCCode(s.code()), s.msg()};
}

}  // namespace gml
