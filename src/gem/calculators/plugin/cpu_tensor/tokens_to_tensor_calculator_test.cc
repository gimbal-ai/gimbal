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

#include "gmock/gmock.h"
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/cpu_tensor/scores_to_classification_calculator.h"
#include "src/gem/exec/core/data_type.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::gem::exec::cpu_tensor::ExecutionContext;
using ::gml::gem::testing::CalculatorTester;

using ::gml::internal::api::core::v1::Classification;

constexpr char kTokensToTensorCalculatorNode[] = R"pbtxt(
calculator: "TokensToTensorCalculator"
input_stream: "TOKENS:tokens"
output_stream: "TENSOR:tensor"
input_side_packet: "EXEC_CTX:cpu_exec_ctx"
)pbtxt";

struct TokensToTensorCalculatorTestCase {
  std::vector<int> tokens;
  std::vector<int> expected_shape;
};

class TokensToTensorCalculatorTest
    : public ::testing::TestWithParam<TokensToTensorCalculatorTestCase> {};

TEST_P(TokensToTensorCalculatorTest, CorrectClassification) {
  auto cpu_exec_ctx = std::make_unique<ExecutionContext>();
  CalculatorTester tester(kTokensToTensorCalculatorNode);

  const auto& test_case = GetParam();

  auto ts = mediapipe::Timestamp::Min();
  auto tensor = tester.WithExecutionContext(cpu_exec_ctx.get())
                    .ForInput("TOKENS", test_case.tokens, ts)
                    .Run()
                    .Result<CPUTensorPtr>("TENSOR", 0);

  ASSERT_THAT(tensor->Shape(), ::testing::ElementsAreArray(test_case.expected_shape));
  auto* tokens = tensor->TypedData<DataType::INT32>();
  for (size_t i = 0; i < test_case.tokens.size(); ++i) {
    EXPECT_EQ(tokens[i], test_case.tokens[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(TokensToTensorCalculatorTests, TokensToTensorCalculatorTest,
                         ::testing::Values(
                             TokensToTensorCalculatorTestCase{
                                 .tokens = {1, 2, 3, 10},
                                 .expected_shape = {1, 4},
                             },
                             TokensToTensorCalculatorTestCase{
                                 .tokens = {10, 190, 20000, 10, 10, 0, 200000},
                                 .expected_shape = {1, 7},
                             }));

}  // namespace gml::gem::calculators::cpu_tensor
