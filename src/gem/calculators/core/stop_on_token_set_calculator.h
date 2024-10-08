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

#include "src/api/corepb/v1/mediastream.pb.h"

namespace gml::gem::calculators::core {

/**
 *  StopOnTokenSetCalculator Graph API:
 *
 *  Calculator stops generation when any of a set of tokens are output.
 *
 *  Inputs:
 *    TOKENS std::vector<int> generated tokens to look for EOS in.
 *
 *  Outputs:
 *    EOS bool
 */
class StopOnTokenSetCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  int64_t max_tokens_before_eos_;
  int64_t tokens_since_eos_;
  absl::flat_hash_set<int64_t> eos_tokens_;
};

}  // namespace gml::gem::calculators::core
