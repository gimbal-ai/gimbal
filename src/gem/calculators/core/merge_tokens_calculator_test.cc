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
#include <mediapipe/framework/port/parse_text_proto.h>

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
  void SetUp() override {
    graph_ = std::make_unique<mediapipe::CalculatorGraph>();

    auto graph_config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(kGraph);

    mediapipe::tool::AddVectorSink("loop_start", &graph_config, &loop_start_packets_);
    mediapipe::tool::AddVectorSink("merged_stream", &graph_config, &merged_stream_packets_);
    mediapipe::tool::AddVectorSink("prompt_timestamp", &graph_config, &prompt_timestamp_packets_);

    ASSERT_OK(graph_->Initialize(graph_config));
    ASSERT_OK(graph_->StartRun({}));
  }

  void InitTimestampDomains(int input_ts, int token_ts) {
    input_ts_ = input_ts;
    token_ts_ = token_ts;
  }

  void AddInputs(const std::vector<int>& input_stream) {
    ASSERT_OK(graph_->AddPacketToInputStream(
        "input_stream",
        mediapipe::MakePacket<std::vector<int>>(input_stream).At(mediapipe::Timestamp(input_ts_))));

    ASSERT_OK(graph_->WaitUntilIdle());
    ++input_ts_;
  }

  void AddOutputs(const std::vector<int>& output_stream, bool eos) {
    ASSERT_OK(graph_->AddPacketToInputStream("output_stream",
                                             mediapipe::MakePacket<std::vector<int>>(output_stream)
                                                 .At(mediapipe::Timestamp(token_ts_))));
    ASSERT_OK(graph_->AddPacketToInputStream(
        "output_eos", mediapipe::MakePacket<bool>(eos).At(mediapipe::Timestamp(token_ts_))));

    ASSERT_OK(graph_->WaitUntilIdle());
    ++token_ts_;
  }

  static constexpr char kGraph[] = R"pbtxt(
    input_stream: "input_stream"
    input_stream: "output_stream"
    input_stream: "output_eos"
    output_stream: "loop_start"
    output_stream: "merged_stream"
    output_stream: "prompt_timestamp"
    node {
      calculator: "MergeTokensCalculator"
      input_stream: "INPUT_TOKENS:input_stream"
      input_stream: "OUTPUT_TOKENS:output_stream"
      input_stream: "OUTPUT_EOS:output_eos"

      output_stream: "LOOP_START:loop_start"
      output_stream: "MERGED_TOKENS:merged_stream"
      output_stream: "PROMPT_TIMESTAMP:prompt_timestamp"
    }
  )pbtxt";

  std::unique_ptr<mediapipe::CalculatorGraph> graph_;
  std::vector<mediapipe::Packet> loop_start_packets_;
  std::vector<mediapipe::Packet> merged_stream_packets_;
  std::vector<mediapipe::Packet> prompt_timestamp_packets_;

  int input_ts_ = 0;
  int token_ts_ = 0;
};

TEST_F(MergeTokensCalculatorTest, MergesTokens) {
  InitTimestampDomains(100, 1000);

  AddInputs({0, 1});
  AddOutputs({1, 1, 1}, false);
  AddOutputs({1, 1, 2}, false);
  AddInputs({0, 2});
  AddInputs({0, 3});
  AddOutputs({1, 1, 3}, true);
  AddOutputs({1, 2, 1}, false);
  AddOutputs({1, 2, 2}, true);
  AddOutputs({1, 3, 1}, true);

  ASSERT_OK(graph_->CloseAllPacketSources());
  ASSERT_OK(graph_->WaitUntilDone());

  std::vector<ExpectedOutput> expected_output = {
      {.prompt_timestamp = 100, .is_loop_start = true, .tokens = {0, 1}},
      {.prompt_timestamp = 100, .is_loop_start = false, .tokens = {1, 1, 1}},
      {.prompt_timestamp = 100, .is_loop_start = false, .tokens = {1, 1, 2}},
      {.prompt_timestamp = 101, .is_loop_start = true, .tokens = {0, 2}},
      {.prompt_timestamp = 101, .is_loop_start = false, .tokens = {1, 2, 1}},
      {.prompt_timestamp = 102, .is_loop_start = true, .tokens = {0, 3}},
  };

  EXPECT_EQ(merged_stream_packets_.size(), expected_output.size());
  for (size_t i = 0; i < merged_stream_packets_.size(); ++i) {
    EXPECT_EQ(merged_stream_packets_[i].Get<std::vector<int>>(), expected_output[i].tokens);
  }

  EXPECT_EQ(loop_start_packets_.size(), expected_output.size());
  for (size_t i = 0; i < loop_start_packets_.size(); ++i) {
    EXPECT_EQ(loop_start_packets_[i].Get<bool>(), expected_output[i].is_loop_start);
  }

  EXPECT_EQ(prompt_timestamp_packets_.size(), expected_output.size());
  for (size_t i = 0; i < prompt_timestamp_packets_.size(); ++i) {
    EXPECT_EQ(prompt_timestamp_packets_[i].Get<mediapipe::Timestamp>().Microseconds(),
              expected_output[i].prompt_timestamp);
  }
}

}  // namespace gml::gem::calculators::core
