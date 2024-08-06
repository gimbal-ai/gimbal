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
#include <mediapipe/framework/stream_handler/sync_set_input_stream_handler.pb.h>
#include <mediapipe/util/packet_test_util.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/logging.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

TEST(MergeTokensCalculatorTest, MergesTokens) {
  constexpr char kGraph[] = R"pbtxt(
    calculator: "MergeTokensCalculator"
    input_stream: "INPUT_TOKENS:input_stream"
    input_stream: "OUTPUT_TOKENS:output_stream"
    input_stream: "OUTPUT_EOS:output_eos"

    output_stream: "LOOP_START:loop_start"
    output_stream: "MERGED_TOKENS:merged_stream"

    input_stream_handler {
      input_stream_handler: "SyncSetInputStreamHandler"
      options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
          sync_set {
            tag_index: "INPUT_TOKENS"
          }
          sync_set {
            tag_index: "OUTPUT_TOKENS"
            tag_index: "OUTPUT_EOS"
          }
        }
      }
    }
  )pbtxt";

  mediapipe::CalculatorRunner runner(kGraph);

  int ts = 0;

  {
    std::vector<int> input_stream = {0, 1};
    runner.MutableInputs()
        ->Tag("INPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(input_stream).At(mediapipe::Timestamp(++ts)));
  }

  {
    std::vector<int> output_stream = {1, 1, 1};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(false).At(mediapipe::Timestamp(ts)));
  }

  {
    std::vector<int> output_stream = {1, 1, 2};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(false).At(mediapipe::Timestamp(ts)));
  }

  {
    std::vector<int> input_stream = {0, 2};
    runner.MutableInputs()
        ->Tag("INPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(input_stream).At(mediapipe::Timestamp(++ts)));
  }

  {
    std::vector<int> input_stream = {0, 3};
    runner.MutableInputs()
        ->Tag("INPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(input_stream).At(mediapipe::Timestamp(++ts)));
  }

  {
    std::vector<int> output_stream = {1, 1, 3};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(true).At(mediapipe::Timestamp(ts)));
  }

  {
    std::vector<int> output_stream = {1, 2, 1};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(false).At(mediapipe::Timestamp(ts)));
  }

  {
    std::vector<int> output_stream = {1, 2, 2};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(true).At(mediapipe::Timestamp(ts)));
  }

  {
    std::vector<int> output_stream = {1, 3, 1};
    runner.MutableInputs()
        ->Tag("OUTPUT_TOKENS")
        .packets.push_back(
            mediapipe::MakePacket<std::vector<int>>(output_stream).At(mediapipe::Timestamp(++ts)));
    runner.MutableInputs()
        ->Tag("OUTPUT_EOS")
        .packets.push_back(mediapipe::MakePacket<bool>(true).At(mediapipe::Timestamp(ts)));
  }

  ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 2);

  std::vector<std::pair<std::vector<int>, bool>> expected_output = {
      {{0, 1}, true}, {{1, 1, 1}, false}, {{1, 1, 2}, false},
      {{0, 2}, true}, {{1, 2, 1}, false}, {{0, 3}, true},
  };

  const std::vector<mediapipe::Packet>& output_tokens = outputs.Tag("MERGED_TOKENS").packets;
  const std::vector<mediapipe::Packet>& output_start = outputs.Tag("LOOP_START").packets;

  EXPECT_EQ(output_tokens.size(), expected_output.size());
  for (size_t i = 0; i < output_tokens.size(); ++i) {
    EXPECT_EQ(output_tokens[i].Get<std::vector<int>>(), expected_output[i].first);
    EXPECT_EQ(output_tokens[i].Timestamp().Microseconds(), i + 1);

    EXPECT_EQ(output_start[i].Get<bool>(), expected_output[i].second);
    EXPECT_EQ(output_start[i].Timestamp().Microseconds(), i + 1);
  }
}

}  // namespace gml::gem::calculators::core
