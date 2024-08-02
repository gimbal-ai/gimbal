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

#include "src/gem/calculators/plugin/huggingface/tokenizer_decoder_calculator.h"

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::huggingface {

using gml::gem::exec::core::Tokenizer;

class FakeTokenizer : public Tokenizer {
 public:
  FakeTokenizer() = default;

  std::vector<int> Encode(const std::string&) const override { return std::vector<int>{1, 2, 3}; }

  std::string Decode(const std::vector<int>&) const override { return "this is a test"; }
};

TEST(TokenizerDecoderCalculatorTest, UsesTokenizerInContext) {
  constexpr char kTokenizerDecoderNode[] = R"pbtxt(
    calculator: "TokenizerDecoderCalculator"
    input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
    input_stream: "TOKEN_IDS:ids"
    output_stream: "DECODED_TOKENS:tokens"
  )pbtxt";

  auto tokenizer = std::make_unique<FakeTokenizer>();
  auto model = gml::gem::exec::huggingface::Model(std::move(tokenizer));
  auto tokenizer_exec_ctx = std::make_unique<gml::gem::exec::huggingface::ExecutionContext>(&model);

  testing::CalculatorTester tester(kTokenizerDecoderNode);
  tester.WithExecutionContext(tokenizer_exec_ctx.get())
      .ForInput("TOKEN_IDS", std::vector<int>{1, 2, 3}, mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<std::string>("DECODED_TOKENS", mediapipe::Timestamp(0), "this is a test");
}

}  // namespace gml::gem::calculators::huggingface
