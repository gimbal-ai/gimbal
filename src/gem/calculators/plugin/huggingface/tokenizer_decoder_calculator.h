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

#include <mediapipe/framework/calculator_framework.h>

#include "src/gem/exec/plugin/huggingface/context.h"

namespace gml::gem::calculators::huggingface {

/**
 * TokenizerDecoderCalculator Graph API:
 *  Input Side Packets:
 *   ExecutionContext tagged with EXEC_CTX
 *  Inputs:
 *   TOKEN_IDS std::vector<int> token IDs to decode.
 *  Outputs:
 *   DECODED_TOKENS std::string decoded tokens.
 **/
class TokenizerDecoderCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
};

}  // namespace gml::gem::calculators::huggingface