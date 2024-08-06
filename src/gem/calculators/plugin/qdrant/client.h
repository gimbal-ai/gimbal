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

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "third_party/github.com/qdrant/go-client/qdrant/points_service.grpc.pb.h"
#include <absl/container/flat_hash_map.h>
#include <grpcpp/grpcpp.h>

#include "src/common/base/statusor.h"

namespace gml::gem::calculators::qdrant {

class QdrantClient {
 public:
  using Document =
      absl::flat_hash_map<std::string, std::variant<std::string, int64_t, double, bool>>;

  explicit QdrantClient(const std::shared_ptr<grpc::Channel>& channel);
  StatusOr<std::vector<Document>> SearchPoints(const std::string& collection_name,
                                               const std::vector<float>& query_vector, int limit);

 private:
  std::unique_ptr<::qdrant::Points::Stub> stub_;
};

}  // namespace gml::gem::calculators::qdrant
