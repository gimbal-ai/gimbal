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
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::gem::exec::cpu_tensor::ExecutionContext;
using ::gml::gem::testing::CalculatorTester;

constexpr char kTensorToTokensCalculatorNode[] = R"pbtxt(
calculator: "TensorToTokensCalculator"
input_stream: "TENSOR:tensor"
output_stream: "TOKENS:tokens"
)pbtxt";

struct TensorToTokensCalculatorTestCase {
  std::vector<int> shape;
  std::vector<int> tokens;
};

class TensorToTokensCalculatorTest
    : public ::testing::TestWithParam<TensorToTokensCalculatorTestCase> {};

TEST_P(TensorToTokensCalculatorTest, CorrectClassification) {
  auto cpu_exec_ctx = std::make_unique<ExecutionContext>();
  CalculatorTester tester(kTensorToTokensCalculatorNode);

  const auto& test_case = GetParam();

  TensorShape shape;
  shape.insert(shape.end(), test_case.shape.begin(), test_case.shape.end());
  ASSERT_OK_AND_ASSIGN(auto tensor, cpu_exec_ctx->TensorPool()->GetTensor(shape, DataType::INT32));

  auto* token_data = tensor->template TypedData<DataType::INT32>();
  for (size_t i = 0; i < test_case.tokens.size(); ++i) {
    token_data[i] = test_case.tokens[i];
  }

  auto ts = mediapipe::Timestamp::Min();
  auto tokens = tester.ForInput("TENSOR", tensor, ts).Run().Result<std::vector<int>>("TOKENS", 0);

  EXPECT_EQ(tokens.size(), test_case.tokens.size());
  EXPECT_EQ(tokens, test_case.tokens);
}

INSTANTIATE_TEST_SUITE_P(TensorToTokensCalculatorTests, TensorToTokensCalculatorTest,
                         ::testing::Values(
                             TensorToTokensCalculatorTestCase{
                                 .shape = {1, 4},
                                 .tokens = {1, 2, 3, 10},
                             },
                             TensorToTokensCalculatorTestCase{
                                 .shape = {1, 7},
                                 .tokens = {10, 190, 20000, 10, 10, 0, 200000},
                             }));

}  // namespace gml::gem::calculators::cpu_tensor
