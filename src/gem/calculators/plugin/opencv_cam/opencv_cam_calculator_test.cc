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

#include "src/gem/calculators/plugin/opencv_cam/opencv_cam_calculator.h"

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/port/parse_text_proto.h>

#include "src/common/bazel/runfiles.h"
#include "src/common/testing/testing.h"

DEFINE_bool(output_frame, false, "Whether to output the actual frame to a file");

constexpr std::string_view kVideoSourceFilename =
    "src/gem/calculators/plugin/opencv_cam/testdata/single_frame.mp4";
constexpr std::string_view kFrameOutputFilename =
    "src/gem/calculators/plugin/opencv_cam/testdata/single_frame.tiff";

namespace {
const std::filesystem::path GetTempDir() {
  // TEST_TMPDIR is set by Bazel when running tests.
  // TODO(vihang): We should probably move this into a common utility function similar to
  // bazel/runfiles.h
  const auto test_tmpdir = getenv("TEST_TMPDIR");
  if (test_tmpdir == nullptr) {
    return std::filesystem::path("/tmp");
  }
  return std::filesystem::path(test_tmpdir);
}
}  // namespace

namespace gml::gem::calculators::opencv_cam {
class OpenCVCamSourceCalculatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_ = std::make_unique<mediapipe::CalculatorGraph>();

    auto source = bazel::RunfilePath(kVideoSourceFilename);
    auto graph_config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
        absl::Substitute(kGraph, source.string()));

    ASSERT_OK(graph_->Initialize(graph_config));
  }

  static constexpr char kGraph[] = R"pbtxt(
    output_stream: "image_frame"
    node {
      calculator: "OpenCVCamSourceCalculator"
      output_stream: "image_frame"
      node_options {
        [type.googleapis.com/gml.gem.calculators.opencv_cam.optionspb.OpenCVCamSourceCalculatorOptions] {
          device_filename: "$0"
        }
      }
    }
  )pbtxt";

  std::unique_ptr<mediapipe::CalculatorGraph> graph_;
};

TEST_F(OpenCVCamSourceCalculatorTest, CaptureImage) {
  bool seen = false;
  cv::Mat frame;

  ASSERT_OK(graph_->ObserveOutputStream("image_frame", [&](const mediapipe::Packet& packet) {
    // Only capture the first frame.
    if (!seen) {
      frame = mediapipe::formats::MatView(&packet.Get<mediapipe::ImageFrame>());
      seen = true;
    }
    return graph_->CloseAllPacketSources();
  }));
  ASSERT_OK(graph_->StartRun({}));
  ASSERT_OK(graph_->WaitUntilDone());

  ASSERT_TRUE(seen);

  cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
  if (FLAGS_output_frame) {
    auto output = GetTempDir() / "output.tiff";
    LOG(INFO) << absl::Substitute("Expected image written to $0", output.string());
    cv::imwrite(output.string(), frame);
  }

  cv::Mat expected = cv::imread(bazel::RunfilePath(kFrameOutputFilename).string());
  cv::Mat diff;
  cv::subtract(frame, expected, diff);

  ASSERT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

}  // namespace gml::gem::calculators::opencv_cam
