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

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/common/testing/testing.h"

#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

namespace gml {
namespace gem {
namespace calculators {
namespace argus {

constexpr char kGraph[] = R"pb(
  calculator: "NvBufSurfToImageFrameCalculator"
  input_stream: "nvbufsurface"
  output_stream: "image_frame"
)pb";

// Expect to find the Y, U and V planes in three files.
// Currently, these are set to a location in /tmp because of how the
// test is run manually in a container on the device.
// TODO(oazizi): Package the data in the image container, and fix these paths.
std::string_view kBufYFilename = "/tmp/buf_y";
std::string_view kBufUFilename = "/tmp/buf_u";
std::string_view kBufVFilename = "/tmp/buf_v";

std::vector<uint8_t> LoadImagePlane(std::string_view filename) {
  std::ifstream plane_fstream(filename, std::ios::in | std::ios::binary);

  LOG(INFO) << absl::Substitute("Loading image plane: $0", filename);

  CHECK(plane_fstream.is_open());

  plane_fstream.seekg(0, std::ios::end);
  CHECK(!plane_fstream.fail());
  std::streampos plane_fstream_size = plane_fstream.tellg();
  CHECK_NE(plane_fstream_size, -1);
  plane_fstream.seekg(0, std::ios::beg);
  CHECK(!plane_fstream.fail());

  std::vector<uint8_t> plane_buf;
  plane_buf.reserve(plane_fstream_size);
  plane_buf.insert(plane_buf.begin(), std::istream_iterator<uint8_t>(plane_fstream),
                   std::istream_iterator<uint8_t>());

  return plane_buf;
}

// This function populates parameters that match the testdata files (buf_u, buf_y, buf_v).
void PopulateNvBufSurface(NvBufSurface* surf, NvBufSurfaceParams* surf_params, uint8_t* y_plane_ptr,
                          uint8_t* u_plane_ptr, uint8_t* v_plane_ptr) {
  constexpr int kPlaneY = 0;
  constexpr int kPlaneU = 1;
  constexpr int kPlaneV = 2;

  surf_params->width = 1280;
  surf_params->height = 720;
  surf_params->pitch = 1280;
  surf_params->colorFormat = NVBUF_COLOR_FORMAT_YUV420;
  surf_params->layout = NVBUF_LAYOUT_PITCH;
  surf_params->dataSize = 1835008;

  surf_params->planeParams.num_planes = 3;
  surf_params->planeParams.width[kPlaneY] = 1280;
  surf_params->planeParams.height[kPlaneY] = 720;
  surf_params->planeParams.pitch[kPlaneY] = 1280;
  surf_params->planeParams.offset[kPlaneY] = 0;  // Irrelevant
  surf_params->planeParams.psize[kPlaneY] = 1048576;
  surf_params->planeParams.bytesPerPix[kPlaneY] = 1;
  surf_params->mappedAddr.addr[kPlaneY] = y_plane_ptr;

  surf_params->planeParams.width[kPlaneU] = 640;
  surf_params->planeParams.height[kPlaneU] = 360;
  surf_params->planeParams.pitch[kPlaneU] = 640;
  surf_params->planeParams.offset[kPlaneU] = 0;  // Irrelevant
  surf_params->planeParams.psize[kPlaneU] = 393216;
  surf_params->planeParams.bytesPerPix[kPlaneU] = 1;
  surf_params->mappedAddr.addr[kPlaneU] = u_plane_ptr;

  surf_params->planeParams.width[kPlaneV] = 640;
  surf_params->planeParams.height[kPlaneV] = 360;
  surf_params->planeParams.pitch[kPlaneV] = 640;
  surf_params->planeParams.offset[kPlaneV] = 0;  // Irrelevant
  surf_params->planeParams.psize[kPlaneV] = 393216;
  surf_params->planeParams.bytesPerPix[kPlaneV] = 1;
  surf_params->mappedAddr.addr[kPlaneV] = v_plane_ptr;

  surf->gpuId = 0;
  surf->batchSize = 1;
  surf->numFilled = 1;
  surf->memType = NVBUF_MEM_DEFAULT;
  surf->surfaceList = surf_params;
}

TEST(NvBufSurfToImageFrameCalculator, conversion) {
  using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

  // Prepare an input image.
  std::vector<uint8_t> y_plane_buf = LoadImagePlane(kBufYFilename);
  std::vector<uint8_t> u_plane_buf = LoadImagePlane(kBufUFilename);
  std::vector<uint8_t> v_plane_buf = LoadImagePlane(kBufVFilename);

  NvBufSurface nvbufsurface;
  NvBufSurfaceParams surf_params[0];
  PopulateNvBufSurface(&nvbufsurface, &(surf_params[0]), y_plane_buf.data(), u_plane_buf.data(),
                       v_plane_buf.data());

  ASSERT_OK_AND_ASSIGN(auto nvbuf_surf,
                       NvBufSurfaceWrapper::TestOnlyCreatePlaceholder(&nvbufsurface));

  nvbuf_surf->DumpInfo();

  mediapipe::CalculatorRunner runner(kGraph);

  // Run the calculator for a single packet.
  mediapipe::Packet p = mediapipe::Adopt(nvbuf_surf.release()).At(mediapipe::Timestamp(1000));
  runner.MutableInputs()->Index(0).packets.push_back(p);

  EXPECT_EQ(runner.Outputs().NumEntries(), 1);

  LOG(INFO) << "Running graph.";

  auto s = runner.Run();
  ASSERT_TRUE(s.ok());

  // Check output.
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  const std::vector<mediapipe::Packet>& output_packets = outputs.Index(0).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& output_image = output_packets[0].Get<mediapipe::ImageFrame>();

  EXPECT_EQ(output_image.Width(), 1280);
  EXPECT_EQ(output_image.Height(), 720);
  EXPECT_EQ(output_image.ChannelSize(), 1);
  EXPECT_EQ(output_image.NumberOfChannels(), 3);
  EXPECT_EQ(output_image.ByteDepth(), 1);
  EXPECT_EQ(output_image.PixelDataSize(), 1280 * 720 * 3);

  // TODO(oazizi): The test above doesn't really check that the conversion has been done properly.
  //               Need some way of actually checking the image pixel data.

  // Something about these data structures doesn't let the test terminate.
  // TODO(oazizi): Investigate further.
  surf_params[0] = {};
  nvbufsurface = {};
}

}  // namespace argus
}  // namespace calculators
}  // namespace gem
}  // namespace gml
