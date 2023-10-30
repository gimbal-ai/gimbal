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

#include "src/gem/exec/core/control_context.h"
extern "C" {
#include <libavcodec/avcodec.h>
}

#include <mediapipe/framework/formats/yuv_image.h>

#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"
#include "src/gem/calculators/plugin/ffmpeg/ffmpeg_video_encoder_calculator.h"
#include "src/gem/exec/core/planar_image.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml {
namespace gem {
namespace calculators {
namespace ffmpeg {

static constexpr char kFFmpegVideoEncoderNode[] = R"pbtxt(
calculator: "FFmpegVideoEncoderCalculator"
input_side_packet: "FRAME_RATE:frame_rate"
input_stream: "PLANAR_IMAGE:planar_image"
output_stream: "AV_PACKETS:av_packets"
)pbtxt";

TEST(FFmpegVideoEncoderCalculator, runs_without_error) {
  testing::CalculatorTester tester(kFFmpegVideoEncoderNode);

  auto yuv_image = std::make_shared<mediapipe::YUVImage>();

  // TODO(james): refactor into common utility to test against a real image.

  // Code for initializing YUVImage taken from mediapipe/util/image_frame_util.cc.
  const int width = 640;
  const int height = 380;
  const int uv_width = (width + 1) / 2;
  const int uv_height = (height + 1) / 2;
  // Align y_stride and uv_stride on 16-byte boundaries.
  const int y_stride = (width + 15) & ~15;
  const int uv_stride = (uv_width + 15) & ~15;
  const int y_size = y_stride * height;
  const int uv_size = uv_stride * uv_height;
  uint8_t* data = reinterpret_cast<uint8_t*>(std::malloc(y_size + uv_size * 2));
  std::function<void()> deallocate = [data]() { std::free(data); };

  uint8_t* y = data;
  uint8_t* u = y + y_size;
  uint8_t* v = u + uv_size;

  for (int i = 0; i < y_size; i++) {
    y[i] = 1;
  }
  for (int i = 0; i < uv_size; i++) {
    u[i] = 2;
    v[i] = 3;
  }
  yuv_image->Initialize(libyuv::FOURCC_I420, deallocate, y, y_stride, u, uv_stride, v, uv_stride,
                        width, height);
  ASSERT_OK_AND_ASSIGN(auto planar_image,
                       exec::core::PlanarImageFor<mediapipe::YUVImage>::Create(
                           std::move(yuv_image), exec::core::ImageFormat::YUV_I420));

  // This test currently only asserts that theres no error running a single frame through the
  // encoder.
  tester.ForInputSidePacket("FRAME_RATE", 30)
      .ForInput("PLANAR_IMAGE", std::move(planar_image), 0)
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

}  // namespace ffmpeg
}  // namespace calculators
}  // namespace gem
}  // namespace gml
