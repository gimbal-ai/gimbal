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

#include "src/gem/exec/core/planar_image.h"
#include <mediapipe/framework/formats/yuv_image.h>
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/testdata/test_image.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {

TEST(PlanarImageFor, YUVImage) {
  auto yuv_image_for_conversion = std::make_shared<mediapipe::YUVImage>();
  testing::LoadTestImageAsYUVImage(yuv_image_for_conversion.get());
  ASSERT_OK_AND_ASSIGN(
      auto planar, PlanarImageFor<mediapipe::YUVImage>::Create(std::move(yuv_image_for_conversion),
                                                               ImageFormat::YUV_I420));

  auto expected_yuv_image = std::make_unique<mediapipe::YUVImage>();
  testing::LoadTestImageAsYUVImage(expected_yuv_image.get());

  EXPECT_EQ(expected_yuv_image->width(), planar->Width());
  EXPECT_EQ(expected_yuv_image->height(), planar->Height());
  EXPECT_EQ(ImageFormat::YUV_I420, planar->Format());

  ASSERT_EQ(3, planar->Planes().size());
  EXPECT_EQ(expected_yuv_image->stride(0), planar->Planes()[0].row_stride);
  EXPECT_EQ(expected_yuv_image->stride(1), planar->Planes()[1].row_stride);
  EXPECT_EQ(expected_yuv_image->stride(2), planar->Planes()[2].row_stride);

  size_t y_size = expected_yuv_image->stride(0) * expected_yuv_image->height();
  size_t uv_size = expected_yuv_image->stride(1) * (expected_yuv_image->height() / 2);

  EXPECT_EQ(y_size, planar->Planes()[0].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[1].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[2].bytes);

  for (size_t plane_idx = 0; plane_idx < planar->Planes().size(); ++plane_idx) {
    const auto* expected_data = expected_yuv_image->data(plane_idx);
    const auto* actual_data = planar->Planes()[plane_idx].data;
    size_t width = expected_yuv_image->width();
    size_t height = expected_yuv_image->height();
    size_t stride = expected_yuv_image->stride(plane_idx);
    if (plane_idx != 0) {
      width = (width + 1) / 2;
      height = (height + 1) / 2;
    }
    for (size_t row_idx = 0; row_idx < height; ++row_idx) {
      const auto* expected_row = &expected_data[row_idx * stride];
      const auto* actual_row = &actual_data[row_idx * stride];
      EXPECT_EQ(0, std::memcmp(expected_row, actual_row, width))
          << absl::Substitute("Plane $0 row $1 differs", plane_idx, row_idx);
    }
  }
}

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
