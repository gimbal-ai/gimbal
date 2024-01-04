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

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <mediapipe/framework/formats/yuv_image.h>

#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"
#include "src/gem/calculators/plugin/ffmpeg/ffmpeg_video_encoder_calculator.h"
#include "src/gem/exec/core/control_context.h"
#include "src/gem/exec/core/planar_image.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/testdata/test_image.h"

namespace gml::gem::calculators::ffmpeg {

static constexpr char kFFmpegVideoEncoderNode[] = R"pbtxt(
calculator: "FFmpegVideoEncoderCalculator"
input_stream: "VIDEO_HEADER:video_header"
input_stream: "PLANAR_IMAGE:planar_image"
output_stream: "AV_PACKETS:av_packets"
)pbtxt";

TEST(FFmpegVideoEncoderCalculator, RunsWithoutError) {
  testing::CalculatorTester tester(kFFmpegVideoEncoderNode);

  auto yuv_image = std::make_shared<mediapipe::YUVImage>();
  testing::LoadTestImageAsYUVImage(yuv_image.get());

  auto video_header = mediapipe::VideoHeader();
  video_header.height = yuv_image->height();
  video_header.width = yuv_image->width();
  video_header.frame_rate = 30;

  ASSERT_OK_AND_ASSIGN(auto planar_image,
                       exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                           std::move(yuv_image), exec::core::ImageFormat::YUV_I420));

  // This test currently only asserts that there's no error running a single frame through the
  // encoder.
  tester.ForInput("VIDEO_HEADER", video_header, mediapipe::Timestamp::PreStream())
      .ForInput("PLANAR_IMAGE", std::move(planar_image), mediapipe::Timestamp::Min())
      .Run();

  const auto& out_packets =
      tester.Result<std::vector<std::unique_ptr<AVPacketWrapper>>>("AV_PACKETS", 0);

  for (const auto& packet : out_packets) {
    LOG(INFO) << "Got AVPacket of size: " << packet->packet()->size;

    size_t nal_units = 0;
    for (int i = 0; i < packet->packet()->size - 3; ++i) {
      uint8_t* bytes = &packet->packet()->data[i];
      if (bytes[0] == 0x00 && bytes[1] == 0x00 && bytes[2] == 0x01) {
        nal_units++;

        int nal_unit_type = bytes[3] & 0x1f;
        LOG(INFO) << "NAL Unit Type: " << nal_unit_type;
      }
    }
    LOG(INFO) << absl::Substitute("Found $0 NAL units in AVPacket", nal_units);
  }
}

}  // namespace gml::gem::calculators::ffmpeg
