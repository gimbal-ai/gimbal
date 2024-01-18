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

#include "src/gem/calculators/core/detections_to_mediapipe_calculator.h"

#include <mediapipe/framework/formats/detection.pb.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kDetectionsToMediapipeNode = R"pbtxt(
calculator: "DetectionsToMediapipeCalculator"
input_stream: "DETECTIONS:detection_list"
output_stream: "DETECTIONS:mp_detection_list"
)pbtxt";

struct DetectionsToMediapipeTestCase {
  std::vector<std::string> detection_pbtxts;
  std::vector<std::string> expected_mp_detection_pbtxts;
};

class DetectionsToMediapipeTest : public ::testing::TestWithParam<DetectionsToMediapipeTestCase> {};

TEST_P(DetectionsToMediapipeTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  auto config = absl::Substitute(kDetectionsToMediapipeNode);
  testing::CalculatorTester tester(config);

  std::vector<Detection> detections(test_case.detection_pbtxts.size());

  for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                              &detections[i]));
  }

  std::vector<mediapipe::Detection> expected_mp_detections(
      test_case.expected_mp_detection_pbtxts.size());
  for (size_t i = 0; i < test_case.expected_mp_detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
        test_case.expected_mp_detection_pbtxts[i], &expected_mp_detections[i]));
  }

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("DETECTIONS", std::move(detections), ts)
      .Run()
      .ExpectOutput<std::vector<mediapipe::Detection>>(
          "DETECTIONS", 0, ts,
          ::testing::Pointwise(::gml::testing::proto::EqProto(), expected_mp_detections));
}

INSTANTIATE_TEST_SUITE_P(DetectionsToMediapipeTestSuite, DetectionsToMediapipeTest,
                         ::testing::Values(DetectionsToMediapipeTestCase{
                             {
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
                             },
                             {
                                 R"pbtxt(
label: "bottle"
score: 0.9
location_data {
  format: LOCATION_FORMAT_RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.45
    ymin: 0.1
    width: 0.1
    height: 0.2
  }
}
)pbtxt",
                             },
                         }));

}  // namespace gml::gem::calculators::core
