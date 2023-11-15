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

#include <fstream>
#include <iostream>

#include <nvbufsurface.h>

#include "src/common/testing/testing.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml::gem::calculators::args {

using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

// Expect to find the Y, U and V planes in three files.
constexpr std::string_view kBufYFilename = "src/gem/calculators/plugin/argus/testdata/buf_y";
constexpr std::string_view kBufUFilename = "src/gem/calculators/plugin/argus/testdata/buf_u";
constexpr std::string_view kBufVFilename = "src/gem/calculators/plugin/argus/testdata/buf_v";

TEST(PlanarImageFor, NvBufSurfaceWrapper) {
  // Prepare an input image.
  ASSERT_OK_AND_ASSIGN(std::string y_plane_buf,
                       gml::ReadFileToString(testing::BazelRunfilePath(kBufYFilename)));
  ASSERT_OK_AND_ASSIGN(std::string u_plane_buf,
                       gml::ReadFileToString(testing::BazelRunfilePath(kBufUFilename)));
  ASSERT_OK_AND_ASSIGN(std::string v_plane_buf,
                       gml::ReadFileToString(testing::BazelRunfilePath(kBufVFilename)));

  NvBufSurface* nvbufsurface;

  constexpr int kImageWidth = 1280;
  constexpr int kImageHeight = 720;

  NvBufSurfaceCreateParams create_params;
  create_params.width = kImageWidth;
  create_params.height = kImageHeight;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_DEFAULT;

  NvBufSurfaceCreate(&nvbufsurface, 1, &create_params);

  ASSERT_OK_AND_ASSIGN(auto nvbuf_surf, NvBufSurfaceWrapper::Create(nvbufsurface));
  ASSERT_OK(nvbuf_surf->MapForCpu());

  nvbuf_surf->DumpInfo();

  ASSERT_OK_AND_ASSIGN(auto planar,
                       gem::exec::core::PlanarImageFor<NvBufSurfaceWrapper>::Create(
                           std::move(nvbuf_surf), gem::exec::core::ImageFormat::YUV_I420));

  EXPECT_EQ(kImageWidth, planar->Width());
  EXPECT_EQ(kImageHeight, planar->Height());
  EXPECT_EQ(gem::exec::core::ImageFormat::YUV_I420, planar->Format());

  ASSERT_EQ(3, planar->Planes().size());

  // WARNING: the lines below are device-specific expectations.
  // NvBufSurfaceCreate automatically uses a larger pitch/stride for the U and V planes when we
  // request the image above, but the pitch may end up being different on other devices.
  // Since this test only runs on the Jetsons for now, we can adjust the expectation when needed.
  EXPECT_EQ(1280, planar->Planes()[0].row_stride);
  EXPECT_EQ(768, planar->Planes()[1].row_stride);
  EXPECT_EQ(768, planar->Planes()[2].row_stride);
}

}  // namespace gml::gem::calculators::args
