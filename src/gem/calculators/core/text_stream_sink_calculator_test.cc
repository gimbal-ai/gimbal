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

#include <absl/container/flat_hash_map.h>
#include <gmock/gmock.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::TextBatch;

struct TextStreamSinkTestCase {
  std::string text;
  bool eos;
};

class TextStreamSinkCalculatorTest : public ::testing::TestWithParam<TextStreamSinkTestCase> {};

TEST_P(TextStreamSinkCalculatorTest, SendsCallback) {
  auto packet_and_expectation = GetParam();

  constexpr char kTextStreamSinkNode[] = R"pbtxt(
    calculator: "TextStreamSinkCalculator"
    input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
    input_stream: "TEXT_BATCH:text"
    input_stream: "EOS:eos"
  )pbtxt";

  mediapipe::CalculatorRunner runner(kTextStreamSinkNode);

  bool callbackCalled = false;
  exec::core::ControlExecutionContext::MediaStreamCallback cb =
      [&](const std::vector<std::unique_ptr<google::protobuf::Message>>& messages) {
        callbackCalled = true;
        EXPECT_EQ(messages.size(), 1);
        auto type = messages[0]->GetTypeName();
        EXPECT_EQ(type, TextBatch::descriptor()->full_name());
        auto text_batch = static_cast<const TextBatch&>(*messages[0]);
        EXPECT_EQ(text_batch.text(), packet_and_expectation.text);
        EXPECT_EQ(text_batch.eos(), packet_and_expectation.eos);

        return Status::OK();
      };

  exec::core::ControlExecutionContext control_ctx;
  control_ctx.RegisterMediaStreamCallback(cb);

  runner.MutableSidePackets()->Get("EXEC_CTX", 0) =
      mediapipe::MakePacket<exec::core::ExecutionContext*>(&control_ctx);

  mediapipe::Packet p = mediapipe::MakePacket<std::string>(packet_and_expectation.text);
  p = p.At(mediapipe::Timestamp(static_cast<int64_t>(0)));
  runner.MutableInputs()->Tag("TEXT_BATCH").packets.push_back(p);

  mediapipe::Packet p2 = mediapipe::MakePacket<bool>(packet_and_expectation.eos);
  p2 = p2.At(mediapipe::Timestamp(static_cast<int64_t>(0)));
  runner.MutableInputs()->Tag("EOS").packets.push_back(p2);

  ASSERT_OK(runner.Run());

  EXPECT_TRUE(callbackCalled);
}

INSTANTIATE_TEST_SUITE_P(TextStreamSinkCalculatorTestSuite, TextStreamSinkCalculatorTest,
                         ::testing::Values(TextStreamSinkTestCase{
                             "this is my text",
                             true,
                         }));

}  // namespace gml::gem::calculators::core
