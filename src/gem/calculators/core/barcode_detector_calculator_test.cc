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
