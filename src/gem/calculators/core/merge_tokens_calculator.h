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
 *  MergeTokensCalculator Graph API:
 *  Enables merging of input stream with the LLM output stream. If the LLM output stream has an eos,
 *  it skips that particular batch of the stream since we don't need to feed that back into the LLM.
 *
 *  Note that this is expected to be used with a SyncSetInputStreamHandler on each of the input
 *  streams. Also note that this calculator uses loop internal timestamps. To get real timestamps,
 *  wrap outputs with a ClockTimestampCalculator.
 *
 *
 *  Inputs:
 *    INPUT_TOKENS std::vector<int> - Token IDs encoded from the input stream.
 *    OUTPUT_TOKENS std::vector<int> - Token IDs to be written to the output stream.
 *    OUTPUT_EOS bool - A flag indicating whether it is the end of the LLM output.
 *
 *  Outputs:
 *    LOOP_START bool - A flag indicating the start of the loop, will be emitted with the first
 *    input token.
 *    MERGED_TOKENS std::vector<int> - Merged stream of tokens.
 *
 */
class MergeTokensCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  void EmitInput(mediapipe::CalculatorContext* cc);
  void HandleInput(mediapipe::CalculatorContext* cc);

  void HandleOutput(mediapipe::CalculatorContext* cc);

  mediapipe::Timestamp internal_timestamp_ = mediapipe::Timestamp(0);
  std::deque<std::vector<int>> input_buffer_;
  bool is_processing_output_ = false;
};

}  // namespace gml::gem::calculators::core
