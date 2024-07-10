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
