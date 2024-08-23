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

#include <gmock/gmock.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/execution_context_calculator.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::TextBatch;

struct TestInput {
  std::string text;
  bool eos;
  mediapipe::Timestamp prompt_ts;
};

class TextStreamSinkCalculatorTest : public ::testing::TestWithParam<std::vector<TestInput>> {
 protected:
  void SetUp() override {
    runner_ = std::make_unique<mediapipe::CalculatorRunner>(kGraph);

    runner_->MutableSidePackets()->Get("EXEC_CTX", 0) =
        mediapipe::MakePacket<exec::core::ExecutionContext*>(&control_ctx_);
  }

  static constexpr char kGraph[] = R"pbtxt(
    calculator: "TextStreamSinkCalculator"
    input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
    input_stream: "TEXT_BATCH:text"
    input_stream: "EOS:eos"
    input_stream: "PROMPT_TIMESTAMP:prompt_ts"
    output_stream: "FINISHED:finished"
  )pbtxt";

  std::unique_ptr<mediapipe::CalculatorRunner> runner_;
  exec::core::ControlExecutionContext control_ctx_;
};

TEST_F(TextStreamSinkCalculatorTest, Basic) {
  auto input_seq = std::vector<TestInput>{TestInput{
                                              "this",
                                              false,
                                              mediapipe::Timestamp(100),
                                          },
                                          TestInput{
                                              "is",
                                              false,
                                              mediapipe::Timestamp(100),
                                          },
                                          TestInput{
                                              "a",
                                              false,
                                              mediapipe::Timestamp(100),
                                          },
                                          TestInput{
                                              "hammer",
                                              true,
                                              mediapipe::Timestamp(100),
                                          },
                                          TestInput{
                                              "Yay!",
                                              true,
                                              mediapipe::Timestamp(101),
                                          }};

  std::vector<std::string> cb_text;
  std::vector<bool> cb_eos;

  exec::core::ControlExecutionContext::MediaStreamCallback cb =
      [&](const std::vector<std::unique_ptr<google::protobuf::Message>>& messages) {
        EXPECT_EQ(messages.size(), 1);
        auto type = messages[0]->GetTypeName();
        EXPECT_EQ(type, TextBatch::descriptor()->full_name());
        auto text_batch = static_cast<const TextBatch&>(*messages[0]);

        cb_text.push_back(text_batch.text());
        cb_eos.push_back(text_batch.eos());

        return Status::OK();
      };

  control_ctx_.RegisterMediaStreamCallback(cb);

  int t = 0;
  for (const auto& input : input_seq) {
    auto ts = mediapipe::Timestamp(static_cast<int64_t>(t));

    auto pkt_text = mediapipe::MakePacket<std::string>(input.text);
    runner_->MutableInputs()->Tag("TEXT_BATCH").packets.push_back(pkt_text.At(ts));

    auto pkt_eos = mediapipe::MakePacket<bool>(input.eos);
    runner_->MutableInputs()->Tag("EOS").packets.push_back(pkt_eos.At(ts));

    auto pkt_prompt_ts = mediapipe::MakePacket<mediapipe::Timestamp>(input.prompt_ts);
    runner_->MutableInputs()->Tag("PROMPT_TIMESTAMP").packets.push_back(pkt_prompt_ts.At(ts));

    ++t;
  }

  ASSERT_OK(runner_->Run());

  EXPECT_THAT(cb_text, ::testing::ElementsAreArray({"this", "is", "a", "hammer", "Yay!"}));
  EXPECT_THAT(cb_eos, ::testing::ElementsAreArray({false, false, false, true, true}));

  const auto& outputs = runner_->Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  const std::vector<mediapipe::Packet>& finished = outputs.Tag("FINISHED").packets;

  EXPECT_EQ(finished.size(), 2);

  EXPECT_EQ(finished[0].Get<bool>(), true);
  EXPECT_EQ(finished[0].Timestamp(), mediapipe::Timestamp(100));

  EXPECT_EQ(finished[1].Get<bool>(), true);
  EXPECT_EQ(finished[1].Timestamp(), mediapipe::Timestamp(101));
}

TEST(TextStreamSinkCalculatorValidatorTest, FinishedRequiresPromptTimestamp) {
  static constexpr char kGraph[] = R"pbtxt(
    calculator: "TextStreamSinkCalculator"
    input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
    input_stream: "TEXT_BATCH:text"
    input_stream: "EOS:eos"
    output_stream: "FINISHED:finished"
  )pbtxt";

  mediapipe::CalculatorRunner runner(kGraph);

  exec::core::ControlExecutionContext control_ctx;
  runner.MutableSidePackets()->Get("EXEC_CTX", 0) =
      mediapipe::MakePacket<exec::core::ExecutionContext*>(&control_ctx);

  absl::Status status = runner.Run();
  ASSERT_TRUE(absl::IsInternal(status));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("PROMPT_TIMESTAMP required when using FINISHED"));
}

}  // namespace gml::gem::calculators::core
