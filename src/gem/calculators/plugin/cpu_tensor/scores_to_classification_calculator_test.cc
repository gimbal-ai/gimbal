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

#include "src/gem/calculators/plugin/cpu_tensor/scores_to_classification_calculator.h"

#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::gem::exec::cpu_tensor::ExecutionContext;
using ::gml::gem::testing::CalculatorTester;

using ::gml::internal::api::core::v1::Classification;

constexpr char kScoresToClassificationNode[] = R"pbtxt(
calculator: "ScoresToClassificationCalculator"
input_stream: "SCORES:scores"
output_stream: "CLASSIFICATION:classification"
node_options {
  [type.googleapis.com/gml.gem.calculators.cpu_tensor.optionspb.ScoresToClassificationCalculatorOptions] {
    index_to_label: "bicycle"
    index_to_label: "car"
    index_to_label: "motorcycle"
  }
}
)pbtxt";

struct ScoresToClassificationTestCase {
  std::vector<float> scores;
  std::string expected_proto_str;
};

class ScoresToClassificationTest : public ::testing::TestWithParam<ScoresToClassificationTestCase> {
};

TEST_P(ScoresToClassificationTest, CorrectClassification) {
  auto cpu_exec_ctx = std::make_unique<ExecutionContext>();
  CalculatorTester tester(kScoresToClassificationNode);

  const auto& test_case = GetParam();

  ASSERT_OK_AND_ASSIGN(auto scores_tensor, cpu_exec_ctx->TensorPool()->GetTensor(
                                               {1, static_cast<int>(test_case.scores.size())},
                                               exec::core::DataType::FLOAT32));
  auto data = reinterpret_cast<float*>(scores_tensor->data());
  std::copy(test_case.scores.begin(), test_case.scores.end(), data);

  Classification expected_classification;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.expected_proto_str,
                                                            &expected_classification));

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("SCORES", std::move(scores_tensor), ts)
      .Run()
      .ExpectOutput<Classification>("CLASSIFICATION", ts,
                                    ::gml::testing::proto::EqProtoMsg(expected_classification));
}

INSTANTIATE_TEST_SUITE_P(
    ScoresToClassificationTests, ScoresToClassificationTest,
    ::testing::Values(
        ScoresToClassificationTestCase{
            .scores = {0, 0.1, 0},
            .expected_proto_str = R"pb(label { label: "car" score: 0.1 })pb"},
        ScoresToClassificationTestCase{
            .scores = {0.5, 0.1, 0.2},
            .expected_proto_str = R"pb(label { label: "bicycle" score: 0.5 })pb"},
        ScoresToClassificationTestCase{
            .scores = {0.1, 0.2, 0.3},
            .expected_proto_str = R"pb(label { label: "motorcycle" score: 0.3 })pb"}));

}  // namespace gml::gem::calculators::cpu_tensor
