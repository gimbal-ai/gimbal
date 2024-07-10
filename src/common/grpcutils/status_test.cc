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
