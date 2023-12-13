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

#include <memory>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"
#include "src/gem/calculators/plugin/ffmpeg/overlayed_ffmpeg_video_sink_calculator.h"

namespace gml::gem::calculators::ffmpeg {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::H264Chunk;
using ::gml::internal::api::core::v1::ImageOverlayChunk;

constexpr std::string_view kDetectionsTag = "DETECTIONS";
constexpr std::string_view kAVPacketTag = "AV_PACKETS";

// TODO(james): move this into calculator options
constexpr size_t kMaxDesiredChunkSize = 512UL * 1024UL;

constexpr float kChunkSizeFudgeFactor = 0.1f;
constexpr size_t kMaxChunkSize =
    static_cast<size_t>((1.0f - kChunkSizeFudgeFactor) * kMaxDesiredChunkSize);

namespace {

// DetectionsToImageOverlayChunks splits the detection list into ImageOverlayChunks. It is not
// responsible for setting the frame number or EOF on the chunks.
Status DetectionsToImageOverlayChunks(const std::vector<Detection>& detections,
                                      std::vector<ImageOverlayChunk>* image_overlay_chunks) {
  // These are estimates for the encoded proto size. See `src/api/corepb/v1/mediastream.proto`.
  // The image overlay chunk has a int64 frame_number and bool eof in addition to each bounding box.
  constexpr size_t kImageOverlayChunkOverhead = sizeof(int64_t) + sizeof(bool);
  constexpr size_t kBoundingBoxSize = 4 * sizeof(float);
  auto chunk = std::make_unique<ImageOverlayChunk>();

  size_t chunk_size = kImageOverlayChunkOverhead;
  for (const auto& detection : detections) {
    // Bounding boxes have 4 floats.
    size_t detection_size = kBoundingBoxSize;
    for (const auto& label : detection.label()) {
      detection_size += sizeof(float) + label.label().size();
    }

    if (chunk->mutable_detections()->detection_size() > 0 &&
        chunk_size + detection_size > kMaxChunkSize) {
      image_overlay_chunks->emplace_back(*chunk);
      chunk = std::make_unique<ImageOverlayChunk>();
      chunk_size = kImageOverlayChunkOverhead;
    }

    (*chunk->mutable_detections()->add_detection()) = detection;
    chunk_size += detection_size;
  }
  if (chunk->mutable_detections()->detection_size() > 0) {
    image_overlay_chunks->emplace_back(*chunk);
  }
  return Status::OK();
}

Status AVPacketsToH264Chunks(const std::vector<std::unique_ptr<AVPacketWrapper>>& packets,
                             std::vector<H264Chunk>* h264_chunks) {
  // This is part of the estimate for the encoded proto size. See
  // `src/api/corepb/v1/mediastream.proto`. The h264 chunk has a int64 frame_number and bool eof in
  // addition to the video data.
  constexpr size_t kH264ChunkOverhead = sizeof(int64_t) + sizeof(bool);
  auto chunk = std::make_unique<H264Chunk>();

  size_t chunk_size = kH264ChunkOverhead;
  for (const auto& packet : packets) {
    auto* av_packet = packet->packet();
    if (chunk->mutable_nal_data()->size() > 0 && chunk_size + av_packet->size > kMaxChunkSize) {
      h264_chunks->emplace_back(*chunk);
      chunk = std::make_unique<H264Chunk>();
      chunk_size = kH264ChunkOverhead;
    }

    chunk->mutable_nal_data()->append(reinterpret_cast<const char*>(av_packet->data),
                                      av_packet->size);
    chunk_size += av_packet->size;
  }
  if (chunk->nal_data().size() > 0) {
    h264_chunks->emplace_back(*chunk);
  }
  return Status::OK();
}

}  // namespace

absl::Status OverlayedFFmpegVideoSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(core::ControlExecutionContextCalculator::UpdateContract(cc));
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  cc->Inputs().Tag(kAVPacketTag).Set<std::vector<std::unique_ptr<AVPacketWrapper>>>();
  return absl::OkStatus();
}

Status OverlayedFFmpegVideoSinkCalculator::ProcessImpl(
    mediapipe::CalculatorContext* cc, exec::core::ControlExecutionContext* control_ctx) {
  std::vector<ImageOverlayChunk> image_overlay_chunks;
  std::vector<H264Chunk> h264_chunks;

  if (cc->Inputs().HasTag(kDetectionsTag) && !cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    const auto& detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    GML_RETURN_IF_ERROR(DetectionsToImageOverlayChunks(detections, &image_overlay_chunks));
  }

  const auto& av_packets =
      cc->Inputs().Tag(kAVPacketTag).Get<std::vector<std::unique_ptr<AVPacketWrapper>>>();
  GML_RETURN_IF_ERROR(AVPacketsToH264Chunks(av_packets, &h264_chunks));

  if (av_packets.empty()) {
    return Status::OK();
  }
  auto frame_number = av_packets[0]->packet()->pts;
  for (size_t i = 0; i < image_overlay_chunks.size(); ++i) {
    image_overlay_chunks[i].set_frame_number(frame_number);
    if (i == (image_overlay_chunks.size() - 1)) {
      image_overlay_chunks[i].set_eof(true);
    }
  }

  for (size_t i = 0; i < h264_chunks.size(); i++) {
    h264_chunks[i].set_frame_number(frame_number);
    if (i == (h264_chunks.size() - 1)) {
      h264_chunks[i].set_eof(true);
    }
  }

  if (control_ctx->HasVideoWithOverlaysCallback()) {
    GML_RETURN_IF_ERROR(
        control_ctx->GetVideoWithOverlaysCallback()(image_overlay_chunks, h264_chunks));
  }

  return Status::OK();
}

REGISTER_CALCULATOR(OverlayedFFmpegVideoSinkCalculator);

}  // namespace gml::gem::calculators::ffmpeg
