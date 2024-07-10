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

#include "src/gem/exec/core/planar_image.h"

#include <mediapipe/framework/formats/yuv_image.h>

#include "src/common/testing/testing.h"
#include "src/gem/testing/core/testdata/test_image.h"

namespace gml::gem::exec::core {

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

  size_t y_size = static_cast<size_t>(expected_yuv_image->stride(0)) * expected_yuv_image->height();
  size_t uv_size =
      static_cast<size_t>(expected_yuv_image->stride(1)) * (expected_yuv_image->height() / 2);

  EXPECT_EQ(y_size, planar->Planes()[0].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[1].bytes);
  EXPECT_EQ(uv_size, planar->Planes()[2].bytes);

  for (int plane_idx = 0; plane_idx < static_cast<int>(planar->Planes().size()); ++plane_idx) {
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

}  // namespace gml::gem::exec::core
