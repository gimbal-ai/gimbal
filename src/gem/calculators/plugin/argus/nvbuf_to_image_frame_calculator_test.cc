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
