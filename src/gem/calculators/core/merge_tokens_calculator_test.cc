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
#include <mediapipe/framework/calculator_runner.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/logging.h"
#include "src/common/testing/testing.h"

namespace gml::gem::calculators::core {

struct ExpectedOutput {
  int prompt_timestamp;
  bool is_loop_start;
  std::vector<int> tokens;
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
    output_stream: "PROMPT_TIMESTAMP:prompt_timestamp"
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
  ASSERT_EQ(outputs.NumEntries(), 3);

  // This test is not quite right. There should really be two different timestamp domains.
  // Unfortunately, it's not a simple matter of using different timestamp domains for the different
  // inputs, because the way that our mediapipe test is structured is that we send all the inputs up
  // front and then call Run(). If we use different timestamp domains, we can't account for the
  // cause and effect relationship between the outputs and the next set of inputs. In the end, this
  // means that the expected prompt_timestamps below are not sequential as expected.
  // TODO(oazizi): Change the test structure to support different timestamp domains.
  // A good reference may be begin_end_item_loop_calculator_graph_test.cc, specifically the part
  // with SendPacketsOfInts() and WaitUntilIdle().
  std::vector<ExpectedOutput> expected_output = {
      {.prompt_timestamp = 1, .is_loop_start = true, .tokens = {0, 1}},
      {.prompt_timestamp = 1, .is_loop_start = false, .tokens = {1, 1, 1}},
      {.prompt_timestamp = 1, .is_loop_start = false, .tokens = {1, 1, 2}},
      {.prompt_timestamp = 4, .is_loop_start = true, .tokens = {0, 2}},
      {.prompt_timestamp = 4, .is_loop_start = false, .tokens = {1, 2, 1}},
      {.prompt_timestamp = 5, .is_loop_start = true, .tokens = {0, 3}},
  };

  const std::vector<mediapipe::Packet>& output_tokens = outputs.Tag("MERGED_TOKENS").packets;
  const std::vector<mediapipe::Packet>& output_start = outputs.Tag("LOOP_START").packets;
  const std::vector<mediapipe::Packet>& prompt_timestamp = outputs.Tag("PROMPT_TIMESTAMP").packets;

  EXPECT_EQ(output_tokens.size(), expected_output.size());
  for (size_t i = 0; i < output_tokens.size(); ++i) {
    EXPECT_EQ(output_tokens[i].Get<std::vector<int>>(), expected_output[i].tokens);
    EXPECT_EQ(output_tokens[i].Timestamp().Microseconds(), i + 1);

    EXPECT_EQ(output_start[i].Get<bool>(), expected_output[i].is_loop_start);
    EXPECT_EQ(output_start[i].Timestamp().Microseconds(), i + 1);

    EXPECT_EQ(prompt_timestamp[i].Get<mediapipe::Timestamp>().Microseconds(),
              expected_output[i].prompt_timestamp);
    EXPECT_EQ(prompt_timestamp[i].Timestamp().Microseconds(), i + 1);
  }
}

}  // namespace gml::gem::calculators::core
