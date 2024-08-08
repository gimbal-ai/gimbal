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

#include "src/gem/calculators/plugin/qdrant/client.h"

#include <iostream>

#include "src/common/base/error.h"
#include "src/common/base/status.h"
#include "src/common/base/statusor.h"
#include "src/common/grpcutils/status.h"

namespace gml::gem::calculators::qdrant {

QdrantClient::QdrantClient(const std::shared_ptr<grpc::Channel>& channel)
    : stub_(::qdrant::Points::NewStub(channel)) {}

StatusOr<std::vector<QdrantClient::Document>> QdrantClient::SearchPoints(
    const std::string& collection_name, const std::vector<float>& query_vector, int limit) {
  ::qdrant::SearchPoints request;
  request.set_collection_name(collection_name);
  for (float v : query_vector) {
    request.add_vector(v);
  }
  request.set_limit(limit);
  auto p = request.mutable_with_payload();
  p->set_enable(true);

  ::qdrant::SearchResponse response;

  grpc::ClientContext context;

  GML_RETURN_IF_ERROR(stub_->Search(&context, request, &response));

  std::vector<Document> documents;
  for (const auto& point : response.result()) {
    Document doc;
    doc["id"] = point.id().uuid();
    doc["score"] = point.score();
    for (const auto& [key, value] : point.payload()) {
      if (value.has_string_value()) {
        doc[key] = value.string_value();
      } else if (value.has_integer_value()) {
        doc[key] = value.integer_value();
      } else if (value.has_double_value()) {
        doc[key] = value.double_value();
      } else if (value.has_bool_value()) {
        doc[key] = value.bool_value();
      }
    }
    documents.push_back(doc);
  }
  return documents;
}

}  // namespace gml::gem::calculators::qdrant
