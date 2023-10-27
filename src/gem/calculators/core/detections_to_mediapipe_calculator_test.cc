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

#include <mediapipe/framework/formats/image_frame.h>
#include "mediapipe/framework/formats/detection.pb.h"

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/detections_to_mediapipe_calculator.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml {
namespace gem {
namespace calculators {
namespace core {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::ImageDetections;
using ::gml::internal::api::core::v1::NormalizedCenterRect;

static constexpr char kDetectionsToMediapipeNode[] = R"pbtxt(
calculator: "DetectionsToMediapipeCalculator"
input_stream: "image_detections"
output_stream: "mp_detection_list"
)pbtxt";

struct DetectionsToMediapipeTestCase {
  std::string image_detections_pbtxt;
  std::string expected_mp_detection_list_pbtxt;
};

class DetectionsToMediapipeTest : public ::testing::TestWithParam<DetectionsToMediapipeTestCase> {};

TEST_P(DetectionsToMediapipeTest, converts_correctly) {
  auto test_case = GetParam();

  auto config = absl::Substitute(kDetectionsToMediapipeNode);
  testing::CalculatorTester tester(config);

  ImageDetections image_detections;

  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.image_detections_pbtxt,
                                                            &image_detections));

  mediapipe::DetectionList expected_mp_detection_list;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      test_case.expected_mp_detection_list_pbtxt, &expected_mp_detection_list));

  tester.ForInput(0, std::move(image_detections), 0)
      .Run()
      .ExpectOutput<mediapipe::DetectionList>(
          "", 0, 0, ::gml::testing::proto::EqProtoMsg(expected_mp_detection_list));
}

INSTANTIATE_TEST_SUITE_P(DetectionsToMediapipeTestSuite, DetectionsToMediapipeTest,
                         ::testing::Values(DetectionsToMediapipeTestCase{
                             R"pbtxt(
detection {
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
}
)pbtxt",
                             R"pbtxt(
detection {
  label: "bottle"
  score: 0.9
  location_data {
    format: RELATIVE_BOUNDING_BOX
    relative_bounding_box {
      xmin: 0.45
      ymin: 0.1
      width: 0.1
      height: 0.2
    }
  }
}
)pbtxt"}));

}  // namespace core
}  // namespace calculators
}  // namespace gem
}  // namespace gml
