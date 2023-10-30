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

namespace gml {
namespace gem {
namespace calculators {
namespace core {

static constexpr char kImageFrameToYUVPlanarImageNode[] = R"pbtxt(
calculator: "ImageFrameToYUVPlanarImage"
input_stream: "IMAGE_FRAME:image_frame"
output_stream: "YUV_IMAGE:planar_image"
)pbtxt";

TEST(ImageFrameToYUVPlanarImage, converts_correctly) {
  testing::CalculatorTester tester(kImageFrameToYUVPlanarImageNode);

  auto yuv_image = std::make_unique<mediapipe::YUVImage>();
  // Code for initializing YUVImage taken from mediapipe/util/image_frame_util.cc.
  const int width = 100;
  const int height = 200;
  const int uv_width = (width + 1) / 2;
  const int uv_height = (height + 1) / 2;
  // Align y_stride and uv_stride on 16-byte boundaries.
  const int y_stride = (width + 15) & ~15;
  const int uv_stride = (uv_width + 15) & ~15;
  const int y_size = y_stride * height;
  const int uv_size = uv_stride * uv_height;
  std::vector<uint8_t> data(y_size + uv_size * 2);
  std::function<void()> deallocate = []() {};

  uint8_t* y = data.data();
  uint8_t* u = y + y_size;
  uint8_t* v = u + uv_size;

  for (int i = 0; i < y_size; i++) {
    y[i] = 128;
  }
  for (int i = 0; i < uv_size; i++) {
    u[i] = 128;
    v[i] = 128;
  }

  yuv_image->Initialize(libyuv::FOURCC_I420, deallocate, y, y_stride, u, uv_stride, v, uv_stride,
                        width, height);

  mediapipe::ImageFrame image_frame;
  mediapipe::image_frame_util::YUVImageToImageFrame(*yuv_image, &image_frame);

  tester.ForInput("IMAGE_FRAME", std::move(image_frame), 0).Run();

  const auto& planar = tester.Result<std::unique_ptr<exec::core::PlanarImage>>("YUV_IMAGE", 0);

  EXPECT_EQ(width, planar->Width());
  EXPECT_EQ(height, planar->Height());
  EXPECT_EQ(exec::core::ImageFormat::YUV_I420, planar->Format());

  ASSERT_EQ(3, planar->Planes().size());
  EXPECT_EQ(y_stride, planar->Planes()[0].row_stride);
  EXPECT_EQ(uv_stride, planar->Planes()[1].row_stride);
  EXPECT_EQ(uv_stride, planar->Planes()[2].row_stride);

  EXPECT_EQ(y_size, planar->Planes()[0].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[1].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[2].bytes);

  for (size_t i = 0; i < planar->Height(); ++i) {
    for (size_t j = 0; j < planar->Width(); ++j) {
      EXPECT_EQ(128, planar->Planes()[0].data[i * y_stride + j]);
      EXPECT_EQ(128, planar->Planes()[1].data[(i / 2) * uv_stride + (j / 2)]);
      EXPECT_EQ(128, planar->Planes()[2].data[(i / 2) * uv_stride + (j / 2)]);
    }
  }
}

}  // namespace core
}  // namespace calculators
}  // namespace gem
}  // namespace gml
