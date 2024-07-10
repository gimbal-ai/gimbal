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

#include "src/gem/calculators/core/modify_detection_with_regression_calculator.h"

#include <mediapipe/framework/formats/detection.pb.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::Regression;

constexpr std::string_view kModifyDetectionNode = R"pbtxt(
calculator: "ModifyDetectionWithRegressionCalculator"
input_stream: "DETECTION:detection"
input_stream: "REGRESSION:regression"
output_stream: "DETECTION:modified_detection"
)pbtxt";

struct ModifyDetectionWithRegressionTestCase {
  std::string detection_pbtxt;
  std::string regression_pbtxt;
  std::string expected_detection_pbtxt;
};

class ModifyDetectionWithRegressionTest
    : public ::testing::TestWithParam<ModifyDetectionWithRegressionTestCase> {};

TEST_P(ModifyDetectionWithRegressionTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  std::string config(kModifyDetectionNode);
  testing::CalculatorTester tester(config);

  Detection detection;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxt, &detection));

  Regression regression;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(test_case.regression_pbtxt, &regression));

  Detection expected_detection;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.expected_detection_pbtxt,
                                                            &expected_detection));

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("DETECTION", std::move(detection), ts)
      .ForInput("REGRESSION", std::move(regression), ts)
      .Run()
      .ExpectOutput<Detection>("DETECTION", 0, ts,
                               ::gml::testing::proto::EqProtoMsg(expected_detection));
}

INSTANTIATE_TEST_SUITE_P(ModifyDetectionWithRegressionTestSuite, ModifyDetectionWithRegressionTest,
                         ::testing::Values(ModifyDetectionWithRegressionTestCase{
                             R"pbtxt(
label {
  label: "bottle"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                             R"pbtxt(
label: "my_label"
value: 0.05
)pbtxt",
                             R"pbtxt(
label {
  label: "bottle"
  score: 0.9
}
label {
  label: "my_label: 0.05"
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                         }));

}  // namespace gml::gem::calculators::core
