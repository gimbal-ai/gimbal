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

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"
#include "src/gem/calculators/plugin/ffmpeg/overlayed_ffmpeg_video_sink_calculator.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml {
namespace gem {
namespace calculators {
namespace ffmpeg {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::H264Chunk;
using ::gml::internal::api::core::v1::ImageOverlayChunk;

static constexpr char kOverlayedFFmpegVideoSinkNode[] = R"pbtxt(
calculator: "OverlayedFFmpegVideoSinkCalculator"
input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
input_stream: "DETECTIONS:detections"
input_stream: "AV_PACKETS:av_packets"
)pbtxt";

struct FFmpegVideoSinkTestCase {
  std::vector<std::string> detection_pbtxts;
  std::vector<size_t> av_packet_sizes;
  std::vector<std::string> expected_overlay_chunk_pbtxts;
  std::vector<std::vector<size_t>> expected_h264_chunks_ids;
};

class OverlayedFFmpegVideoSinkTest : public ::testing::TestWithParam<FFmpegVideoSinkTestCase> {};

TEST_P(OverlayedFFmpegVideoSinkTest, outputs_expected_chunks) {
  auto test_case = GetParam();

  auto config = absl::Substitute(kOverlayedFFmpegVideoSinkNode);
  testing::CalculatorTester tester(config);

  std::vector<Detection> detections(test_case.detection_pbtxts.size());

  for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                              &detections[i]));
  }

  std::vector<std::unique_ptr<AVPacketWrapper>> packets(test_case.av_packet_sizes.size());
  for (const auto& [i, packet_size] : Enumerate(test_case.av_packet_sizes)) {
    auto packet = AVPacketWrapper::Create();
    auto* av_packet = packet->mutable_packet();
    av_new_packet(av_packet, packet_size);
    std::memset(av_packet->data, static_cast<char>(i), packet_size);
    av_packet->pts = 0;
    packets[i] = std::move(packet);
  }

  std::vector<ImageOverlayChunk> expected_overlay_chunks(
      test_case.expected_overlay_chunk_pbtxts.size());
  for (size_t i = 0; i < test_case.expected_overlay_chunk_pbtxts.size(); ++i) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
        test_case.expected_overlay_chunk_pbtxts[i], &expected_overlay_chunks[i]));
  }
  std::vector<H264Chunk> expected_h264_chunks;
  for (auto& chunk_ids : test_case.expected_h264_chunks_ids) {
    auto& chunk = expected_h264_chunks.emplace_back();
    chunk.set_frame_number(0);
    for (auto id : chunk_ids) {
      auto size = test_case.av_packet_sizes[id];
      chunk.mutable_nal_data()->append(std::string(size, static_cast<char>(id)));
    }
  }
  expected_h264_chunks.back().set_eof(true);

  std::vector<ImageOverlayChunk> actual_image_overlay_chunks;
  std::vector<H264Chunk> actual_h264_chunks;

  exec::core::ControlExecutionContext::VideoWithOverlaysCallback cb =
      [&](const std::vector<ImageOverlayChunk>& image_overlay_chunks,
          const std::vector<H264Chunk>& h264_chunks) {
        actual_image_overlay_chunks.insert(actual_image_overlay_chunks.end(),
                                           image_overlay_chunks.begin(),
                                           image_overlay_chunks.end());
        actual_h264_chunks.insert(actual_h264_chunks.end(), h264_chunks.begin(), h264_chunks.end());
        return Status::OK();
      };

  exec::core::ControlExecutionContext control_ctx;
  control_ctx.RegisterVideoWithOverlaysCallback(cb);

  tester.WithExecutionContext(&control_ctx)
      .ForInput("DETECTIONS", std::move(detections), 0)
      .ForInput("AV_PACKETS", std::move(packets), 0)
      .Run();

  EXPECT_THAT(actual_image_overlay_chunks,
              ::testing::Pointwise(::gml::testing::proto::EqProto(), expected_overlay_chunks));
  EXPECT_THAT(actual_h264_chunks,
              ::testing::Pointwise(::gml::testing::proto::EqProto(), expected_h264_chunks));
}

INSTANTIATE_TEST_SUITE_P(OverlayedFFmpegVideoSinkTestSuite, OverlayedFFmpegVideoSinkTest,
                         ::testing::Values(
                             FFmpegVideoSinkTestCase{
                                 .detection_pbtxts =
                                     {
                                         R"pbtxt(
label {
  label: "bottle"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                                     },
                                 .av_packet_sizes =
                                     {
                                         100,
                                     },
                                 .expected_overlay_chunk_pbtxts =
                                     {
                                         R"pbtxt(
frame_number: 0
eof: true
detections {
  detection {
    label {
      label: "bottle"
      score: 0.9
    }
    bounding_box {
      xc: 0.5
      yc: 0.2
      width: 0.1
      height: 0.2
    }
  }
}
)pbtxt",
                                     },
                                 .expected_h264_chunks_ids =
                                     {
                                         {0},
                                     },
                             },
                             FFmpegVideoSinkTestCase{
                                 .detection_pbtxts =
                                     {
                                         R"pbtxt(
label {
  label: "bottle"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                                         R"pbtxt(
label {
  label: "person"
  score: 0.9
}
bounding_box {
  xc: 0.1
  yc: 0.5
  width: 0.1
  height: 0.5
}
)pbtxt",
                                     },
                                 .av_packet_sizes =
                                     {
                                         1024,
                                         512 * 1024,
                                     },
                                 .expected_overlay_chunk_pbtxts =
                                     {
                                         R"pbtxt(
frame_number: 0
eof: true
detections {
  detection {
    label {
      label: "bottle"
      score: 0.9
    }
    bounding_box {
      xc: 0.5
      yc: 0.2
      width: 0.1
      height: 0.2
    }
  }
  detection {
    label {
      label: "person"
      score: 0.9
    }
    bounding_box {
      xc: 0.1
      yc: 0.5
      width: 0.1
      height: 0.5
    }
  }
}
)pbtxt",
                                     },
                                 .expected_h264_chunks_ids =
                                     {
                                         {0},
                                         // Second packet is large so it should force a new chunk.
                                         {1},
                                     },
                             },
                             FFmpegVideoSinkTestCase{
                                 .detection_pbtxts =
                                     {
                                         R"pbtxt(
label {
  label: "bottle"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                                         R"pbtxt(
label {
  label: "person"
  score: 0.9
}
bounding_box {
  xc: 0.1
  yc: 0.5
  width: 0.1
  height: 0.5
}
)pbtxt",
                                     },
                                 .av_packet_sizes =
                                     {
                                         1024 * 1024,
                                     },
                                 .expected_overlay_chunk_pbtxts =
                                     {
                                         R"pbtxt(
frame_number: 0
eof: true
detections {
  detection {
    label {
      label: "bottle"
      score: 0.9
    }
    bounding_box {
      xc: 0.5
      yc: 0.2
      width: 0.1
      height: 0.2
    }
  }
  detection {
    label {
      label: "person"
      score: 0.9
    }
    bounding_box {
      xc: 0.1
      yc: 0.5
      width: 0.1
      height: 0.5
    }
  }
}
)pbtxt",
                                     },
                                 .expected_h264_chunks_ids =
                                     {
                                         // If the packet is larger than the chunk size it should
                                         // still be output as a chunk with a single packet.
                                         {0},
                                     },
                             }));

}  // namespace ffmpeg
}  // namespace calculators
}  // namespace gem
}  // namespace gml
