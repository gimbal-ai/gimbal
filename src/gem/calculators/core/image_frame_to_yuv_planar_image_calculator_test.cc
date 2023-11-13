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
#include <mediapipe/framework/formats/yuv_image.h>
#include <mediapipe/util/image_frame_util.h>

#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/image_frame_to_yuv_planar_image.h"
#include "src/gem/exec/core/planar_image.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/matchers.h"
#include "src/gem/testing/core/testdata/test_image.h"

namespace gml::gem::calculators::core {

static constexpr char kImageFrameToYUVPlanarImageNode[] = R"pbtxt(
calculator: "ImageFrameToYUVPlanarImage"
input_stream: "IMAGE_FRAME:image_frame"
output_stream: "YUV_IMAGE:planar_image"
)pbtxt";

TEST(ImageFrameToYUVPlanarImage, ConvertsCorrectly) {
  testing::CalculatorTester tester(kImageFrameToYUVPlanarImageNode);

  mediapipe::ImageFrame image_frame;
  testing::LoadTestImageAsImageFrame(&image_frame);

  tester.ForInput("IMAGE_FRAME", std::move(image_frame), 0).Run();

  const auto& planar = tester.Result<std::unique_ptr<exec::core::PlanarImage>>("YUV_IMAGE", 0);

  auto expected_yuv_image = std::make_unique<mediapipe::YUVImage>();
  testing::LoadTestImageAsYUVImage(expected_yuv_image.get());

  ASSERT_OK_AND_ASSIGN(auto expected_planar,
                       exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                           std::move(expected_yuv_image), exec::core::ImageFormat::YUV_I420));

  EXPECT_THAT(planar.get(), testing::PlanarImageEq(expected_planar.get()));
}

}  // namespace gml::gem::calculators::core
