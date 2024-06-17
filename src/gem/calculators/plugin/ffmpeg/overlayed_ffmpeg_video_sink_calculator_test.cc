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

#include <mediapipe/framework/formats/video_stream_header.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"
#include "src/gem/calculators/plugin/ffmpeg/overlayed_ffmpeg_video_sink_calculator.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::ffmpeg {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::H264Chunk;
using ::gml::internal::api::core::v1::ImageHistogramBatch;
using ::gml::internal::api::core::v1::ImageOverlayChunk;
using ::gml::internal::api::core::v1::ImageQualityMetrics;
using ::gml::internal::api::core::v1::Segmentation;
using ::gml::internal::api::core::v1::VideoHeader;

static constexpr char kOverlayedFFmpegVideoSinkNode[] = R"pbtxt(
calculator: "OverlayedFFmpegVideoSinkCalculator"
input_side_packet: "EXEC_CTX:ctrl_exec_ctx"
input_stream: "AV_PACKETS:av_packets"
input_stream: "VIDEO_HEADER:video_header"
$0
)pbtxt";

struct FFmpegVideoSinkTestCase {
  std::vector<std::string> detection_pbtxts;
  std::vector<int> av_packet_sizes;
  std::vector<std::string> expected_overlay_chunk_pbtxts;
  std::vector<std::vector<size_t>> expected_h264_chunks_ids;
  std::optional<std::string> image_hist_batch_pbtxt;
  std::optional<std::string> image_quality_pbtxt;
  std::optional<std::string> segmentation_pbtxt;
};

class OverlayedFFmpegVideoSinkTest : public ::testing::TestWithParam<FFmpegVideoSinkTestCase> {};

TEST_P(OverlayedFFmpegVideoSinkTest, OutputsExpectedChunks) {
  auto test_case = GetParam();

  std::vector<std::string> overlay_input_streams;
  if (test_case.detection_pbtxts.size() > 0) {
    overlay_input_streams.emplace_back(R"pb(input_stream: "DETECTIONS:detections")pb");
  }
  ImageHistogramBatch image_hist_batch;
  if (test_case.image_hist_batch_pbtxt.has_value()) {
    overlay_input_streams.emplace_back(R"pb(input_stream: "IMAGE_HIST:hist")pb");
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
        test_case.image_hist_batch_pbtxt.value(), &image_hist_batch));
  }
  ImageQualityMetrics image_quality;
  if (test_case.image_quality_pbtxt.has_value()) {
    overlay_input_streams.emplace_back(R"pb(input_stream: "IMAGE_QUALITY:quality")pb");
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.image_quality_pbtxt.value(),
                                                              &image_quality));
  }
  Segmentation segmentation;
  if (test_case.segmentation_pbtxt.has_value()) {
    overlay_input_streams.emplace_back(R"pb(input_stream: "SEGMENTATION:segmentation")pb");
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.segmentation_pbtxt.value(),
                                                              &segmentation));
  }

  auto config =
      absl::Substitute(kOverlayedFFmpegVideoSinkNode, absl::StrJoin(overlay_input_streams, "\n"));
  testing::CalculatorTester tester(config);

  std::vector<Detection> detections(test_case.detection_pbtxts.size());

  for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                              &detections[i]));
  }

  auto video_header = mediapipe::VideoHeader();
  video_header.height = 160;
  video_header.width = 210;
  video_header.frame_rate = 12;

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
    chunk.set_frame_ts(0);
    for (auto id : chunk_ids) {
      auto size = test_case.av_packet_sizes[id];
      chunk.mutable_nal_data()->append(std::string(size, static_cast<char>(id)));
    }
  }
  expected_h264_chunks.back().set_eof(true);

  std::vector<ImageOverlayChunk> actual_image_overlay_chunks;
  std::vector<H264Chunk> actual_h264_chunks;

  exec::core::ControlExecutionContext::VideoWithOverlaysCallback cb =
      [&](const std::vector<std::unique_ptr<google::protobuf::Message>>& messages) {
        for (auto& message : messages) {
          auto type = message->GetTypeName();
          if (type == ImageOverlayChunk::descriptor()->full_name()) {
            actual_image_overlay_chunks.emplace_back();
            actual_image_overlay_chunks.back().CopyFrom(
                static_cast<const ImageOverlayChunk&>(*message));
          } else if (type == H264Chunk::descriptor()->full_name()) {
            actual_h264_chunks.emplace_back();
            actual_h264_chunks.back().CopyFrom(static_cast<const H264Chunk&>(*message));
          } else if (type == VideoHeader::descriptor()->full_name()) {
            auto header = static_cast<const VideoHeader&>(*message);
            EXPECT_EQ(header.height(), 160);
            EXPECT_EQ(header.width(), 210);
            EXPECT_EQ(header.frame_rate(), 12);
          }
        }
        return Status::OK();
      };

  exec::core::ControlExecutionContext control_ctx;
  control_ctx.RegisterVideoWithOverlaysCallback(cb);

  tester.WithExecutionContext(&control_ctx)
      .ForInput("VIDEO_HEADER", video_header, mediapipe::Timestamp::PreStream());
  auto ts = mediapipe::Timestamp::Min();
  tester.WithExecutionContext(&control_ctx).ForInput("AV_PACKETS", std::move(packets), ts);
  if (detections.size() > 0) {
    tester.ForInput("DETECTIONS", std::move(detections), ts);
  }
  if (test_case.image_hist_batch_pbtxt.has_value()) {
    tester.ForInput("IMAGE_HIST", std::move(image_hist_batch), ts);
  }
  if (test_case.image_quality_pbtxt.has_value()) {
    tester.ForInput("IMAGE_QUALITY", std::move(image_quality), ts);
  }
  if (test_case.segmentation_pbtxt.has_value()) {
    tester.ForInput("SEGMENTATION", std::move(segmentation), ts);
  }
  tester.Run();

  EXPECT_THAT(actual_image_overlay_chunks,
              ::testing::Pointwise(::gml::testing::proto::EqProto(), expected_overlay_chunks));
  EXPECT_THAT(actual_h264_chunks,
              ::testing::Pointwise(::gml::testing::proto::EqProto(), expected_h264_chunks));
}

auto sink_tests = ::testing::Values(
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
                frame_ts: 0
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
        .image_hist_batch_pbtxt = {},
        .image_quality_pbtxt = {},
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
                frame_ts: 0
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
        .image_hist_batch_pbtxt = {},
        .image_quality_pbtxt = {},
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
                frame_ts: 0
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
        .image_hist_batch_pbtxt = {},
        .image_quality_pbtxt = {},
    },
    FFmpegVideoSinkTestCase{
        .detection_pbtxts = {},
        .av_packet_sizes =
            {
                100,
            },
        .expected_overlay_chunk_pbtxts =
            {
                // Always expect an overlay chunk even if it's empty.
                // This ensures detections are cleared if any.
                R"pbtxt(
                frame_ts: 0
                eof: true
                detections {}
                )pbtxt",
            },
        .expected_h264_chunks_ids =
            {
                {0},
            },
        .image_hist_batch_pbtxt = {},
        .image_quality_pbtxt = {},
    },
    FFmpegVideoSinkTestCase{
        .detection_pbtxts = {},
        .av_packet_sizes =
            {
                100,
            },
        .expected_overlay_chunk_pbtxts =
            {
                R"pbtxt(
                frame_ts: 0
                eof: false
                detections {}
                )pbtxt",
                R"pbtxt(
                frame_ts: 0
                eof: false
                histograms {
                  histograms {
                    min: 0.1
                    max: 0.9
                    num: 100
                    sum: 1000
                    bucket: 0
                    bucket: 10
                    bucket: 90
                  }
                }
                )pbtxt",
                R"pbtxt(
                frame_ts: 0
                eof: true
                image_quality {
                  brisque_score: 0.5
                }
                )pbtxt",
            },
        .expected_h264_chunks_ids =
            {
                {0},
            },
        .image_hist_batch_pbtxt =
            {
                R"pbtxt(
                histograms {
                  min: 0.1
                  max: 0.9
                  num: 100
                  sum: 1000
                  bucket: 0
                  bucket: 10
                  bucket: 90
                }
                )pbtxt",
            },
        .image_quality_pbtxt =
            {
                R"pbtxt(
                brisque_score: 0.5
                )pbtxt",
            },
    },
    FFmpegVideoSinkTestCase{
        .av_packet_sizes =
            {
                100,
            },
        .expected_overlay_chunk_pbtxts =
            {
                R"pbtxt(
                frame_ts: 0
                eof: true
                segmentation {
                  masks {
                    label: "person"
                    run_length_encoding: [0,1,1,1]
                  }
                  masks {
                    label: "cat"
                    run_length_encoding: [1,1,1,0]
                  }
                  width: 640
                  height: 480
                }
                )pbtxt",
            },
        .expected_h264_chunks_ids =
            {
                {0},
            },
        .segmentation_pbtxt =
            {
                R"pbtxt(
          masks {
            label: "person"
            run_length_encoding: [0,1,1,1]
          }
          masks {
            label: "cat"
            run_length_encoding: [1,1,1,0]
          }
          width: 640
          height: 480
          )pbtxt"},
    });

INSTANTIATE_TEST_SUITE_P(OverlayedFFmpegVideoSinkTestSuite, OverlayedFFmpegVideoSinkTest,
                         sink_tests);

}  // namespace gml::gem::calculators::ffmpeg
