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

#include "src/gem/calculators/core/stop_on_token_set_calculator.h"

#include <google/protobuf/text_format.h>

#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/optionspb/stop_on_token_set_calculator_options.pb.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kStopOnTokenSetNode = R"pbtxt(
calculator: "StopOnTokenSetCalculator"
input_stream: "TOKENS:tokens"
output_stream: "EOS:eos"
)pbtxt";

struct StopOnTokenSetTestCase {
  std::vector<int> output_tokens;
  int64_t max_tokens_before_eos;
  std::vector<int> eos_tokens;

  bool expected_eos;
};

std::string NodeOptionsFromTestCase(const StopOnTokenSetTestCase& tc) {
  optionspb::StopOnTokenSetCalculatorOptions options;
  options.set_max_tokens_before_eos(tc.max_tokens_before_eos);
  options.mutable_eos_tokens()->Assign(tc.eos_tokens.begin(), tc.eos_tokens.end());

  mediapipe::CalculatorGraphConfig::Node node;
  node.add_node_options()->PackFrom(options);

  std::string out;
  google::protobuf::TextFormat::PrintToString(node, &out);
  return out;
}

class StopOnTokenSetTest : public ::testing::TestWithParam<StopOnTokenSetTestCase> {};

TEST_P(StopOnTokenSetTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  auto node_options = NodeOptionsFromTestCase(test_case);
  std::string config(absl::StrCat(kStopOnTokenSetNode, node_options));
  testing::CalculatorTester tester(config);

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("TOKENS", std::move(test_case.output_tokens), ts)
      .Run()
      .ExpectOutput<bool>("EOS", 0, ts, test_case.expected_eos);
}

INSTANTIATE_TEST_SUITE_P(StopOnTokenSetTestSuite, StopOnTokenSetTest,
                         ::testing::Values(
                             StopOnTokenSetTestCase{
                                 .output_tokens = {1, 2, 3, 0},
                                 .max_tokens_before_eos = 256,
                                 .eos_tokens = {0, 9, 10},
                                 .expected_eos = true,
                             },
                             StopOnTokenSetTestCase{
                                 .output_tokens = {1, 2, 3, 4, 5, 6},
                                 .max_tokens_before_eos = 4,
                                 .eos_tokens = {0},
                                 .expected_eos = true,
                             },
                             StopOnTokenSetTestCase{
                                 .output_tokens = {1, 2, 3, 4},
                                 .max_tokens_before_eos = 5,
                                 .eos_tokens = {0},
                                 .expected_eos = false,
                             },
                             StopOnTokenSetTestCase{
                                 .output_tokens = {1, 2, 3, 4},
                                 .max_tokens_before_eos = 4,
                                 .eos_tokens = {0},
                                 .expected_eos = true,
                             }));

}  // namespace gml::gem::calculators::core
