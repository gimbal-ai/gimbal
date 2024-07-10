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

#include <limits>

#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/regression_to_proto_options.pb.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPool;
using ::gml::internal::api::core::v1::Regression;

static constexpr char kRegressionToProtoNode[] = R"pbtxt(
calculator: "RegressionToProtoCalculator"
input_stream: "TENSOR:regression_tensor"
output_stream: "REGRESSION:regression"
node_options {
  [type.googleapis.com/gml.gem.calculators.cpu_tensor.optionspb.RegressionToProtoOptions] {
    label: "value"
  }
}
)pbtxt";

TEST(RegressionToProtoCalculatorTest, ConvertsCorrectly) {
  CPUTensorPool pool;
  testing::CalculatorTester tester(kRegressionToProtoNode);

  ASSERT_OK_AND_ASSIGN(auto tensor, pool.GetTensor(TensorShape{1}, DataType::FLOAT32));

  auto* data = tensor->TypedData<DataType::FLOAT32>();
  data[0] = 10.0;

  Regression expected_regression;
  expected_regression.set_label("value");
  expected_regression.set_value(10.0);

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("TENSOR", std::move(tensor), ts)
      .Run()
      .ExpectOutput<Regression>("REGRESSION", 0, ts,
                                gml::testing::proto::EqProtoMsg(expected_regression));
}

}  // namespace gml::gem::calculators::cpu_tensor
