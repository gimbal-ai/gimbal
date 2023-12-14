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

#include "src/gem/calculators/plugin/opencv_cam/opencv_cam_calculator.h"

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>

#include "src/common/testing/testing.h"

namespace gml::gem::calculators::opencv_cam {

static constexpr char kGraph[] = R"pb(
  calculator: "OpenCVCamSourceCalculator"
  output_stream: "image_frame"
  node_options {
    [type.googleapis.com/gml.gem.calculators.opencv_cam.optionspb.OpenCVCamSourceCalculatorOptions] {
      device_filename: "/dev/video0"
      max_num_frames: 1
    }
  }
)pb";

// This test runs the graph to take a single capture from the camera.
// While there are some basic checks, the real validation is that the frame is correct.
// Note however, that the image check is not done in this test, since the camera
// output will be different every time.
// TODO(oazizi): Investigate a loopback device to make the test deterministic.
TEST(OpenCVCamSourceCalculator, CaptureImage) {
  mediapipe::CalculatorRunner runner(kGraph);

  LOG(INFO) << "Running graph.";

  ASSERT_OK(runner.Run());

  // Check output.
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  const std::vector<mediapipe::Packet>& output_packets = outputs.Index(0).packets;
  EXPECT_EQ(output_packets.size(), 1);
  const auto& output_frame = output_packets[0].Get<mediapipe::ImageFrame>();

  const std::string kOutFile = "/tmp/output_frame.jpg";

  LOG(INFO) << absl::Substitute(
      "You can now check that the output $0 is a well-formed image from the camera.", kOutFile);
  cv::imwrite(kOutFile, mediapipe::formats::MatView(&output_frame));
}

}  // namespace gml::gem::calculators::opencv_cam
