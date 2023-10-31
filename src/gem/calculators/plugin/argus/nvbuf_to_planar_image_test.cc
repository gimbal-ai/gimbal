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

#include "src/common/base/file.h"
#include "src/common/testing/testing.h"

#include "src/gem/calculators/plugin/argus/nvbuf_to_planar_image.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml {
namespace gem {
namespace calculators {
namespace argus {

using ::gml::gem::exec::core::ImageFormat;
using ::gml::gem::exec::core::PlanarImageFor;

constexpr char kGraph[] = R"pb(
  calculator: "NvBufSurfToPlanarImageCalculator"
  input_stream: "nvbufsurface"
  output_stream: "image_frame"
)pb";

// Expect to find the Y, U and V planes in three files.
// Currently, these are set to a location in /tmp because of how the
// test is run manually in a container on the device.
// TODO(oazizi): Fix these paths.
char kBufYFilename[] = "/app/testdata/buf_y";
char kBufUFilename[] = "/app/testdata/buf_u";
char kBufVFilename[] = "/app/testdata/buf_v";

TEST(NvBufSurfToPlanarImageCalculator, conversion) {
  using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

  // Prepare an input image.
  ASSERT_OK_AND_ASSIGN(std::string y_plane_buf, gml::ReadFileToString(kBufYFilename));
  ASSERT_OK_AND_ASSIGN(std::string u_plane_buf, gml::ReadFileToString(kBufUFilename));
  ASSERT_OK_AND_ASSIGN(std::string v_plane_buf, gml::ReadFileToString(kBufVFilename));

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
  mediapipe::Packet p = mediapipe::Adopt(nvbuf_surf.release()).At(mediapipe::Timestamp(1000));
  runner.MutableInputs()->Index(0).packets.push_back(p);

  EXPECT_EQ(runner.Outputs().NumEntries(), 1);

  LOG(INFO) << "Running graph.";

  ASSERT_OK(runner.Run());

  // Check output.
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  const std::vector<mediapipe::Packet>& output_packets = outputs.Index(0).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& output_image = output_packets[0].Get<PlanarImageFor<NvBufSurfaceWrapper>>();

  EXPECT_EQ(output_image.Width(), 1280);
  EXPECT_EQ(output_image.Height(), 720);
  EXPECT_EQ(output_image.Format(), ImageFormat::YUV_I420);
}

}  // namespace argus
}  // namespace calculators
}  // namespace gem
}  // namespace gml
