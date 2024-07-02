/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
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
