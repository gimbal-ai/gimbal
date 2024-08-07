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

#include "src/gem/calculators/core/generate_chat_message_calculator.h"

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

TEST(GenerateChatMessageCalculatorTest, SendsCallback) {
  constexpr char kGraph[] = R"pbtxt(
    calculator: "GenerateChatMessageCalculator"
    node_options {
      [type.googleapis.com/gml.gem.calculators.core.optionspb.GenerateChatMessageCalculatorOptions] {
        message_template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        preset_system_prompt: "You are a helpful assistant. Use the following documents to help the user."
      }
    }
    input_stream: "QUERY:query"
    input_stream: "DOCUMENTS:prompts"
    output_stream: "TEXT:text"
  )pbtxt";

  mediapipe::CalculatorRunner runner(kGraph);

  auto p = mediapipe::MakePacket<std::string>("Hello, world!")
               .At(mediapipe::Timestamp(static_cast<int64_t>(0)));
  runner.MutableInputs()->Tag("QUERY").packets.push_back(p);
  std::vector<std::string> docs{"doc1", "doc2", "doc3"};
  auto p2 = mediapipe::MakePacket<std::vector<std::string>>(docs).At(
      mediapipe::Timestamp(static_cast<int64_t>(0)));
  runner.MutableInputs()->Tag("DOCUMENTS").packets.push_back(p2);

  ASSERT_OK(runner.Run());

  // Check output.
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  ASSERT_EQ(outputs.Tag("TEXT").packets[0].Get<std::string>(), R"(<|im_start|>system
You are a helpful assistant. Use the following documents to help the user.<|im_end|>
<|im_start|>system
doc1<|im_end|>
<|im_start|>system
doc2<|im_end|>
<|im_start|>system
doc3<|im_end|>
<|im_start|>user
Hello, world!<|im_end|>
)");
}

}  // namespace gml::gem::calculators::core
