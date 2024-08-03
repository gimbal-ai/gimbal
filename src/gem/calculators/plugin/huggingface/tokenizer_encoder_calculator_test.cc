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

#include "src/gem/calculators/plugin/huggingface/tokenizer_encoder_calculator.h"

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/huggingface/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::huggingface {

TEST(TokenizerEncoderCalculatorTest, UsesTokenizerInContext) {
  constexpr char kTokenizerEncoderNode[] = R"pbtxt(
    calculator: "TokenizerEncoderCalculator"
    input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
    input_stream: "TEXT:text"
    output_stream: "TOKEN_IDS:ids"
  )pbtxt";

  auto text = std::string{"this is a test"};
  auto tokenizer = std::make_unique<FakeTokenizer>(text, std::vector<int>{1, 2, 3});
  auto model = gml::gem::exec::huggingface::Model(std::move(tokenizer));
  auto tokenizer_exec_ctx = std::make_unique<gml::gem::exec::huggingface::ExecutionContext>(&model);

  testing::CalculatorTester tester(kTokenizerEncoderNode);
  tester.WithExecutionContext(tokenizer_exec_ctx.get())
      .ForInput("TEXT", std::string("this is a test"), mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<std::vector<int>>("TOKEN_IDS", mediapipe::Timestamp(0),
                                      std::vector<int>{1, 2, 3});
}

}  // namespace gml::gem::calculators::huggingface
