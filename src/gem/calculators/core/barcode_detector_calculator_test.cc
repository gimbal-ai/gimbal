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

#include "src/gem/calculators/core/barcode_detector_calculator.h"

#include <gmock/gmock.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/matchers.h"
#include "src/gem/testing/core/testdata/test_image.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

static constexpr char kBarcodeDetectorNode[] = R"pbtxt(
calculator: "BarcodeDetectorCalculator"
input_stream: "IMAGE_FRAME:image_frame"
output_stream: "DETECTIONS:detections"
)pbtxt";

TEST(BarcodeDetectorCalculator, DetectsBarcodesDecodesOne) {
  testing::CalculatorTester tester(kBarcodeDetectorNode);

  mediapipe::ImageFrame image_frame;
  testing::LoadTestImageAsImageFrame(&image_frame, testing::TestImageType::WITH_BARCODE);

  tester.ForInput("IMAGE_FRAME", std::move(image_frame), mediapipe::Timestamp::Min()).Run();

  const auto& detections = tester.Result<std::vector<Detection>>("DETECTIONS", 0);

  ASSERT_EQ(3, detections.size());
  std::vector<std::string> decoded;
  for (const auto& detection : detections) {
    for (const auto& label : detection.label()) {
      decoded.emplace_back(absl::StripPrefix(label.label(), "Barcode: "));
    }
  }
  EXPECT_THAT(decoded, ::testing::UnorderedElementsAre("[UPC_A] 812674025261", "<decode failed>",
                                                       "<decode failed>"));
}

}  // namespace gml::gem::calculators::core
