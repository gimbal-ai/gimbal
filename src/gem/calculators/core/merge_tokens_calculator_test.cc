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

#include "src/gem/calculators/core/merge_tokens_calculator.h"

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/logging.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

struct ExpectedOutput {
  std::vector<int> tokens;
  bool is_loop_start;
};

class MergeTokensCalculatorTest : public ::testing::Test {
 protected:
  void SetUp() override { runner_ = std::make_unique<mediapipe::CalculatorRunner>(kGraph); }

  void AddInputs(int ts, const std::vector<int>& input_stream) {
    runner_->MutableInputs()
        ->Tag("INPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(input_stream).At(mediapipe::Timestamp(ts)));
  }

  void AddOutputs(int ts, const std::vector<int>& output_stream, bool eos) {
    runner_->MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(ts)));
    runner_->MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(eos).At(mediapipe::Timestamp(ts)));
  }

  static constexpr char kGraph[] = R"pbtxt(
    calculator: "MergeTokensCalculator"
    input_stream: "INPUT_TOKENS:input_stream"
    input_stream: "OUTPUT_TOKENS:output_stream"
    input_stream: "OUTPUT_EOS:output_eos"

    output_stream: "LOOP_START:loop_start"
    output_stream: "MERGED_TOKENS:merged_stream"
  )pbtxt";

  std::unique_ptr<mediapipe::CalculatorRunner> runner_;
};

TEST_F(MergeTokensCalculatorTest, MergesTokens) {
  int ts = 0;

  AddInputs(++ts, {0, 1});
  AddOutputs(++ts, {1, 1, 1}, false);
  AddOutputs(++ts, {1, 1, 2}, false);
  AddInputs(++ts, {0, 2});
  AddInputs(++ts, {0, 3});
  AddOutputs(++ts, {1, 1, 3}, true);
  AddOutputs(++ts, {1, 2, 1}, false);
  AddOutputs(++ts, {1, 2, 2}, true);
  AddOutputs(++ts, {1, 3, 1}, true);

  ASSERT_OK(runner_->Run());

  const auto& outputs = runner_->Outputs();
  ASSERT_EQ(outputs.NumEntries(), 2);

  std::vector<ExpectedOutput> expected_output = {
      {.tokens = {0, 1}, .is_loop_start = true},     {.tokens = {1, 1, 1}, .is_loop_start = false},
      {.tokens = {1, 1, 2}, .is_loop_start = false}, {.tokens = {0, 2}, .is_loop_start = true},
      {.tokens = {1, 2, 1}, .is_loop_start = false}, {.tokens = {0, 3}, .is_loop_start = true},
  };

  const std::vector<mediapipe::Packet>& output_tokens = outputs.Tag("MERGED_TOKENS").packets;
  const std::vector<mediapipe::Packet>& output_start = outputs.Tag("LOOP_START").packets;

  EXPECT_EQ(output_tokens.size(), expected_output.size());
  for (size_t i = 0; i < output_tokens.size(); ++i) {
    EXPECT_EQ(output_tokens[i].Get<std::vector<int>>(), expected_output[i].tokens);
    EXPECT_EQ(output_tokens[i].Timestamp().Microseconds(), i + 1);

    EXPECT_EQ(output_start[i].Get<bool>(), expected_output[i].is_loop_start);
    EXPECT_EQ(output_start[i].Timestamp().Microseconds(), i + 1);
  }
}

}  // namespace gml::gem::calculators::core
