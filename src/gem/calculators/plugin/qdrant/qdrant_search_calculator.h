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

#include <mediapipe/framework/calculator_framework.h>

#include "src/gem/calculators/plugin/qdrant/client.h"
#include "src/gem/calculators/plugin/qdrant/optionspb/qdrant_search_calculator_options.pb.h"

namespace gml::gem::calculators::qdrant {

using ::gml::gem::calculators::qdrant::QdrantClient;

/**
 * QdrantSearchCalculator Graph API:
 *
 *  Options:
 *    address The address of the Qdrant server.
 *    collection The name of the Qdrant collection.
 *    limit The number of results from Qdrant to fetch.
 *  Inputs:
 *    EMBEDDING CPUTensorPtr The query vector to search for.
 *  Outputs:
 *    DOCUMENTS std::vector<string> The documents returned by Qdrant.
 *
 */
class QdrantSearchCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  absl::Status ConnectToQdrant();

  std::unique_ptr<QdrantClient> client_;
  std::shared_ptr<grpc::Channel> channel_;
  std::string address_;

  std::string collection_name_;
  std::string payload_key_;
  int64_t limit_;
};

}  // namespace gml::gem::calculators::qdrant
