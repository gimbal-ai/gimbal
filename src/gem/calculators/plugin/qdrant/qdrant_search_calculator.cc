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

#include "src/gem/calculators/plugin/qdrant/qdrant_search_calculator.h"

#include <thread>
#include <variant>
#include <vector>

#include <absl/status/status.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/timestamp.h>

#include "src/gem/calculators/plugin/qdrant/client.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

namespace gml::gem::calculators::qdrant {

using ::gml::gem::calculators::qdrant::optionspb::QdrantSearchCalculatorOptions;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;

constexpr std::string_view kEmbeddingTag = "EMBEDDING";
constexpr std::string_view kDocumentsTag = "DOCUMENTS";

absl::Status QdrantSearchCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kEmbeddingTag).Set<CPUTensorPtr>();
  cc->Outputs().Tag(kDocumentsTag).Set<std::vector<std::string>>();

  cc->SetTimestampOffset(0);

  return absl::OkStatus();
}

absl::Status QdrantSearchCalculator::ConnectToQdrant() {
  // Assume SSL for now, until we update the options with more SSL configs.
  auto channel_creds = grpc::SslCredentials(grpc::SslCredentialsOptions());
  channel_ = grpc::CreateChannel(address_, channel_creds);
  client_ = std::make_unique<QdrantClient>(channel_);
  return absl::OkStatus();
}

absl::Status QdrantSearchCalculator::Open(mediapipe::CalculatorContext* cc) {
  auto& options = cc->Options<QdrantSearchCalculatorOptions>();

  address_ = options.address();

  GML_ABSL_RETURN_IF_ERROR(this->ConnectToQdrant());

  collection_name_ = options.collection();
  limit_ = options.limit();
  payload_key_ = options.payload_key();
  return absl::OkStatus();
}

absl::Status QdrantSearchCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (channel_->GetState(true) != GRPC_CHANNEL_READY) {
    VLOG(1) << "Reconnecting to qdrant gRPC server...";
    GML_ABSL_RETURN_IF_ERROR(this->ConnectToQdrant());
  }

  const auto& tensor = cc->Inputs().Tag(kEmbeddingTag).Get<CPUTensorPtr>();

  auto shape = tensor->Shape();
  size_t num_elements = 0;

  // Support single embeddings or embeddings with a batch of size 1.
  if (shape.size() == 1) {
    num_elements = shape[0];
  }
  if (shape.size() == 2 && shape[0] == 1) {
    num_elements = shape[1];
  }

  if (num_elements == 0) {
    return AbslStatusAdapter(
        error::InvalidArgument("incorrect shape passed to QdrantSearchCalculator, expeted a single "
                               "embedding or a batch of size 1, received [$0]",
                               absl::StrJoin(tensor->Shape(), ",")));
  }

  if (tensor->DataType() != exec::core::DataType::FLOAT32) {
    return absl::InvalidArgumentError("Input tensor must be of type FLOAT32");
  }
  const float* float_data = tensor->TypedData<exec::core::DataType::FLOAT32>();
  auto embedding = std::vector<float>(float_data, float_data + num_elements);

  GML_ABSL_ASSIGN_OR_RETURN(auto search_results,
                            client_->SearchPoints(collection_name_, embedding, limit_));

  std::vector<std::string> documents;
  for (const auto& res : search_results) {
    auto it = res.find(payload_key_);
    if (it != res.end()) {
      if (!std::holds_alternative<std::string>(it->second)) {
        LOG_EVERY_N(INFO, 5) << absl::StrFormat("Payload is not a string for key %s", payload_key_);
        continue;
      }
      auto payload = std::get<std::string>(it->second);
      documents.push_back(payload);
    }
  }

  cc->Outputs()
      .Tag(kDocumentsTag)
      .AddPacket(
          mediapipe::MakePacket<std::vector<std::string>>(documents).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

REGISTER_CALCULATOR(QdrantSearchCalculator);

}  // namespace gml::gem::calculators::qdrant
