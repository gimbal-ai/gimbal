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

#include "src/gem/calculators/plugin/argus/nvbuf_to_image_frame_calculator.h"

#include <fstream>
#include <iostream>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/common/base/file.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/testing/testing.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

namespace gml::gem::calculators::argus {

constexpr char kGraph[] = R"pb(
  calculator: "NvBufSurfToImageFrameCalculator"
  input_stream: "nvbufsurface"
  output_stream: "image_frame"
)pb";

// Expect to find the Y, U and V planes in three files.

constexpr std::string_view kBufYFilename = "src/gem/calculators/plugin/argus/testdata/buf_y";
constexpr std::string_view kBufUFilename = "src/gem/calculators/plugin/argus/testdata/buf_u";
constexpr std::string_view kBufVFilename = "src/gem/calculators/plugin/argus/testdata/buf_v";

TEST(NvBufSurfToImageFrameCalculator, conversion) {
  using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

  // Prepare an input image.
  ASSERT_OK_AND_ASSIGN(std::string y_plane_buf,
                       gml::ReadFileToString(bazel::RunfilePath(kBufYFilename)));
  ASSERT_OK_AND_ASSIGN(std::string u_plane_buf,
                       gml::ReadFileToString(bazel::RunfilePath(kBufUFilename)));
  ASSERT_OK_AND_ASSIGN(std::string v_plane_buf,
                       gml::ReadFileToString(bazel::RunfilePath(kBufVFilename)));

  NvBufSurface* nvbufsurface;

  NvBufSurfaceCreateParams create_params;
  create_params.width = 1280;
  create_params.height = 720;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_DEFAULT;

  NvBufSurfaceCreate(&nvbufsurface, 1, &create_params);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<NvBufSurfaceWrapper> nvbuf_surf,
                       NvBufSurfaceWrapper::Create(nvbufsurface));
  ASSERT_OK(nvbuf_surf->MapForCpu());

  nvbuf_surf->DumpInfo();

  mediapipe::CalculatorRunner runner(kGraph);

  // Run the calculator for a single packet.
  mediapipe::Packet p =
      mediapipe::MakePacket<std::shared_ptr<NvBufSurfaceWrapper>>(std::move(nvbuf_surf));
  p = p.At(mediapipe::Timestamp(1000));
  runner.MutableInputs()->Index(0).packets.push_back(p);

  EXPECT_EQ(runner.Outputs().NumEntries(), 1);

  LOG(INFO) << "Running graph.";

  ASSERT_OK(runner.Run());

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
}

}  // namespace gml::gem::calculators::argus
