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

#include "src/gem/calculators/core/modify_detection_with_classification_calculator.h"

#include <mediapipe/framework/formats/detection.pb.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Classification;
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kModifyDetectionNode = R"pbtxt(
calculator: "ModifyDetectionWithClassificationCalculator"
input_stream: "DETECTION:detection"
input_stream: "CLASSIFICATION:classification"
output_stream: "DETECTION:modified_detection"
)pbtxt";

struct ModifyDetectionWithClassificationTestCase {
  std::string detection_pbtxt;
  std::string classification_pbtxt;
  std::string expected_detection_pbtxt;
};

class ModifyDetectionWithClassificationTest
    : public ::testing::TestWithParam<ModifyDetectionWithClassificationTestCase> {};

TEST_P(ModifyDetectionWithClassificationTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  std::string config(kModifyDetectionNode);
  testing::CalculatorTester tester(config);

  Detection detection;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxt, &detection));

  Classification classification;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.classification_pbtxt,
                                                            &classification));

  Detection expected_detection;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.expected_detection_pbtxt,
                                                            &expected_detection));

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("DETECTION", std::move(detection), ts)
      .ForInput("CLASSIFICATION", std::move(classification), ts)
      .Run()
      .ExpectOutput<Detection>("DETECTION", 0, ts,
                               ::gml::testing::proto::EqProtoMsg(expected_detection));
}

INSTANTIATE_TEST_SUITE_P(ModifyDetectionWithClassificationTestSuite,
                         ModifyDetectionWithClassificationTest,
                         ::testing::Values(
                             ModifyDetectionWithClassificationTestCase{
                                 .detection_pbtxt = R"pbtxt(
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
                                 .classification_pbtxt = R"pbtxt(
label {
  label: "my_label"
  score: 0.3
}
)pbtxt",
                                 .expected_detection_pbtxt = R"pbtxt(
label {
  label: "my_label"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                             },
                             ModifyDetectionWithClassificationTestCase{
                                 .detection_pbtxt = R"pbtxt(
label {
  label: "can"
  score: 0.8
}
label {
  label: "bottle"
  score: 0.2
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                                 .classification_pbtxt = R"pbtxt(
label {
  label: "my_label"
  score: 0.3
}
label {
  label: "not_my_label"
  score: 0.2
}
)pbtxt",
                                 .expected_detection_pbtxt = R"pbtxt(
label {
  label: "my_label"
  score: 0.8
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
