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

  tester.ForInput("IMAGE_FRAME", std::move(image_frame), mediapipe::Timestamp::Min()).Run();

  const auto& planar = tester.Result<std::unique_ptr<exec::core::PlanarImage>>("YUV_IMAGE", 0);

  auto expected_yuv_image = std::make_unique<mediapipe::YUVImage>();
  testing::LoadTestImageAsYUVImage(expected_yuv_image.get());

  ASSERT_OK_AND_ASSIGN(auto expected_planar,
                       exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                           std::move(expected_yuv_image), exec::core::ImageFormat::YUV_I420));

  EXPECT_THAT(planar.get(), testing::PlanarImageEq(expected_planar.get()));
}

}  // namespace gml::gem::calculators::core
