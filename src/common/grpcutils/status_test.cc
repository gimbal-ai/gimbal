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

#include "src/common/grpcutils/status.h"

#include "src/common/base/status.h"
#include "src/common/testing/testing.h"

namespace gml {

TEST(StatusAdapter, GRPCStatus) {
  auto grpc_status = ::grpc::Status(::grpc::StatusCode::INTERNAL, "test msg");
  auto gml_status = StatusAdapter(grpc_status);
  EXPECT_EQ(gml::types::CODE_INTERNAL, gml_status.code());
  EXPECT_EQ("test msg", gml_status.msg());
}

}  // namespace gml
