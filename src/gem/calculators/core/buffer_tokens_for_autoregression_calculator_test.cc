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

#include "src/gem/calculators/core/buffer_tokens_for_autoregression_calculator.h"

#include <google/protobuf/text_format.h>

#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kBufferTokensForAutoregressionNode = R"pbtxt(
calculator: "BufferTokensForAutoregressionCalculator"
input_stream: "TOKENS:tokens"
input_stream: "LOOP_START:loop_start"
output_stream: "ALL_TOKENS:all_tokens"
)pbtxt";

struct BufferTokensForAutoregressionTestCase {
  std::vector<std::vector<int>> token_inputs;
  std::vector<bool> loop_starts;

  std::vector<std::vector<int>> expected_all_tokens;
};

class BufferTokensForAutoregressionTest
    : public ::testing::TestWithParam<BufferTokensForAutoregressionTestCase> {};

TEST_P(BufferTokensForAutoregressionTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  std::string config(kBufferTokensForAutoregressionNode);
  testing::CalculatorTester tester(config);

  int64_t ts = 0;
  for (size_t i = 0; i < test_case.token_inputs.size(); ++i) {
    ++ts;
    tester.ForInput("TOKENS", test_case.token_inputs[i], mediapipe::Timestamp(ts));
    tester.ForInput<bool>("LOOP_START", test_case.loop_starts[i], mediapipe::Timestamp(ts));
  }
  tester.Run();

  ts = 0;
  for (const auto& expected : test_case.expected_all_tokens) {
    ++ts;
    tester.ExpectOutput<std::vector<int>>("ALL_TOKENS", 0, mediapipe::Timestamp(ts), expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BufferTokensForAutoregressionTestSuite, BufferTokensForAutoregressionTest,
    ::testing::Values(
        BufferTokensForAutoregressionTestCase{
            .token_inputs = {{1, 2, 3, 4, 5}, {6}, {7}},
            .loop_starts =
                {
                    true,
                    false,
                    false,
                },
            .expected_all_tokens =
                {
                    {1, 2, 3, 4, 5},
                    {1, 2, 3, 4, 5, 6},
                    {1, 2, 3, 4, 5, 6, 7},
                },
        },
        BufferTokensForAutoregressionTestCase{
            .token_inputs = {{1, 2, 3, 4, 5}, {6}, {7}, {10, 11, 12, 13}, {1}, {1}},
            .loop_starts =
                {
                    true,
                    false,
                    false,
                    true,
                    false,
                    false,
                },
            .expected_all_tokens =
                {
                    {1, 2, 3, 4, 5},
                    {1, 2, 3, 4, 5, 6},
                    {1, 2, 3, 4, 5, 6, 7},
                    {10, 11, 12, 13},
                    {10, 11, 12, 13, 1},
                    {10, 11, 12, 13, 1, 1},
                },
        }));

}  // namespace gml::gem::calculators::core
