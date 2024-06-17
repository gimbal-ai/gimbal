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
using ::gml::internal::api::core::v1::ImageHistogramBatch;
using ::gml::internal::api::core::v1::ImageOverlayChunk;
using ::gml::internal::api::core::v1::ImageQualityMetrics;
using ::gml::internal::api::core::v1::Segmentation;
using ::gml::internal::api::core::v1::VideoHeader;

constexpr std::string_view kDetectionsTag = "DETECTIONS";
constexpr std::string_view kAVPacketTag = "AV_PACKETS";
constexpr std::string_view kVideoHeaderTag = "VIDEO_HEADER";
constexpr std::string_view kImageHistTag = "IMAGE_HIST";
constexpr std::string_view kImageQualityTag = "IMAGE_QUALITY";
constexpr std::string_view kFinishedTag = "FINISHED";
constexpr std::string_view kSegmentationTag = "SEGMENTATION";

// TODO(james): move this into calculator options
constexpr size_t kMaxDesiredChunkSize = 512UL * 1024UL;

constexpr float kChunkSizeFudgeFactor = 0.1f;
constexpr size_t kMaxChunkSize =
    static_cast<size_t>((1.0f - kChunkSizeFudgeFactor) * kMaxDesiredChunkSize);

namespace {

// DetectionsToImageOverlayChunks splits the detection list into ImageOverlayChunks.
Status DetectionsToImageOverlayChunks(
    const std::vector<Detection>& detections, int64_t frame_ts,
    std::vector<std::unique_ptr<google::protobuf::Message>>* messages) {
  // These are estimates for the encoded proto size. See `src/api/corepb/v1/mediastream.proto`.
  // The image overlay chunk has a int64 frame_ts and bool eof in addition to each bounding box.
  constexpr size_t kImageOverlayChunkOverhead = sizeof(int64_t) + sizeof(bool);
  constexpr size_t kBoundingBoxSize = 4 * sizeof(float);
  auto chunk = std::make_unique<ImageOverlayChunk>();
  chunk->set_frame_ts(frame_ts);

  size_t chunk_size = kImageOverlayChunkOverhead;
  for (const auto& detection : detections) {
    // Bounding boxes have 4 floats.
    size_t detection_size = kBoundingBoxSize;
    for (const auto& label : detection.label()) {
      detection_size += sizeof(float) + label.label().size();
    }

    if (chunk->mutable_detections()->detection_size() > 0 &&
        chunk_size + detection_size > kMaxChunkSize) {
      messages->push_back(std::move(chunk));
      chunk = std::make_unique<ImageOverlayChunk>();
      chunk->set_frame_ts(frame_ts);
      chunk_size = kImageOverlayChunkOverhead;
    }

    (*chunk->mutable_detections()->add_detection()) = detection;
    chunk_size += detection_size;
  }
  if (chunk->mutable_detections()->detection_size() > 0) {
    messages->push_back(std::move(chunk));
  }
  return Status::OK();
}

Status AVPacketsToH264Chunks(const std::vector<std::unique_ptr<AVPacketWrapper>>& packets,
                             int64_t frame_ts,
                             std::vector<std::unique_ptr<google::protobuf::Message>>* messages) {
  // This is part of the estimate for the encoded proto size. See
  // `src/api/corepb/v1/mediastream.proto`. The h264 chunk has a int64 frame_ts and bool eof in
  // addition to the video data.
  constexpr size_t kH264ChunkOverhead = sizeof(int64_t) + sizeof(bool);

  auto chunk = std::make_unique<H264Chunk>();
  chunk->set_frame_ts(frame_ts);

  size_t chunk_size = kH264ChunkOverhead;
  for (const auto& packet : packets) {
    auto* av_packet = packet->packet();
    if (chunk->mutable_nal_data()->size() > 0 && chunk_size + av_packet->size > kMaxChunkSize) {
      messages->push_back(std::move(chunk));
      chunk = std::make_unique<H264Chunk>();
      chunk->set_frame_ts(frame_ts);
      chunk_size = kH264ChunkOverhead;
    }

    chunk->mutable_nal_data()->append(reinterpret_cast<const char*>(av_packet->data),
                                      av_packet->size);
    chunk_size += av_packet->size;
  }
  if (chunk->nal_data().size() > 0) {
    messages->push_back(std::move(chunk));
  }
  return Status::OK();
}

}  // namespace

absl::Status OverlayedFFmpegVideoSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(core::ControlExecutionContextCalculator::UpdateContract(cc));
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  if (cc->Inputs().HasTag(kImageHistTag)) {
    cc->Inputs().Tag(kImageHistTag).Set<ImageHistogramBatch>();
  }
  if (cc->Inputs().HasTag(kImageQualityTag)) {
    cc->Inputs().Tag(kImageQualityTag).Set<ImageQualityMetrics>();
  }
  if (cc->Inputs().HasTag(kSegmentationTag)) {
    cc->Inputs().Tag(kSegmentationTag).Set<Segmentation>();
  }
  cc->Inputs().Tag(kAVPacketTag).Set<std::vector<std::unique_ptr<AVPacketWrapper>>>();
  cc->Inputs().Tag(kVideoHeaderTag).Set<mediapipe::VideoHeader>();
  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Set<bool>();
  }
  cc->SetProcessTimestampBounds(true);
  return absl::OkStatus();
}

Status OverlayedFFmpegVideoSinkCalculator::ProcessImpl(
    mediapipe::CalculatorContext* cc, exec::core::ControlExecutionContext* control_ctx) {
  std::vector<std::unique_ptr<google::protobuf::Message>> messages;

  if (cc->InputTimestamp() == mediapipe::Timestamp::PreStream()) {
    video_header_ = cc->Inputs().Tag(kVideoHeaderTag).Get<mediapipe::VideoHeader>();
    return Status::OK();
  }

  if (cc->Inputs().Tag(kAVPacketTag).IsEmpty()) {
    if (cc->Outputs().HasTag(kFinishedTag)) {
      cc->Outputs()
          .Tag(kFinishedTag)
          .SetNextTimestampBound(cc->InputTimestamp().NextAllowedInStream());
    }
    return Status::OK();
  }

  const auto& av_packets =
      cc->Inputs().Tag(kAVPacketTag).Get<std::vector<std::unique_ptr<AVPacketWrapper>>>();

  if (av_packets.empty()) {
    if (cc->Outputs().HasTag(kFinishedTag)) {
      cc->Outputs()
          .Tag(kFinishedTag)
          .SetNextTimestampBound(cc->InputTimestamp().NextAllowedInStream());
    }
    return Status::OK();
  }
  auto frame_ts = av_packets[0]->packet()->pts;

  GML_RETURN_IF_ERROR(AVPacketsToH264Chunks(av_packets, frame_ts, &messages));
  // We always have a H264Chunk, since we return early if that's not the case. Hence this static
  // cast is safe.
  static_cast<H264Chunk*>(messages.back().get())->set_eof(true);

  if (cc->Inputs().HasTag(kDetectionsTag) && !cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    const auto& detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    GML_RETURN_IF_ERROR(DetectionsToImageOverlayChunks(detections, frame_ts, &messages));
  } else if (cc->Inputs().HasTag(kSegmentationTag) &&
             !cc->Inputs().Tag(kSegmentationTag).IsEmpty()) {
    const auto& segmentation = cc->Inputs().Tag(kSegmentationTag).Get<Segmentation>();
    // TODO(james): separate segmentation over multiple chunks.
    auto chunk = std::make_unique<ImageOverlayChunk>();
    chunk->set_frame_ts(frame_ts);
    chunk->mutable_segmentation()->MergeFrom(segmentation);
    messages.push_back(std::move(chunk));
  } else {
    // Always include an overlay chunk of type detections. This ensures that we clear
    // stale detections if any.
    auto chunk = std::make_unique<ImageOverlayChunk>();
    chunk->set_frame_ts(frame_ts);
    chunk->mutable_detections();
    messages.push_back(std::move(chunk));
  }

  if (cc->Inputs().HasTag(kImageHistTag) && !cc->Inputs().Tag(kImageHistTag).IsEmpty()) {
    const auto& hist = cc->Inputs().Tag(kImageHistTag).Get<ImageHistogramBatch>();
    auto chunk = std::make_unique<ImageOverlayChunk>();
    chunk->set_frame_ts(frame_ts);
    (*chunk->mutable_histograms()) = hist;
    messages.push_back(std::move(chunk));
  }
  if (cc->Inputs().HasTag(kImageQualityTag) && !cc->Inputs().Tag(kImageQualityTag).IsEmpty()) {
    const auto& quality = cc->Inputs().Tag(kImageQualityTag).Get<ImageQualityMetrics>();
    auto chunk = std::make_unique<ImageOverlayChunk>();
    chunk->set_frame_ts(frame_ts);
    (*chunk->mutable_image_quality()) = quality;
    messages.push_back(std::move(chunk));
  }
  // We always add an overlay chunk, so we can be certain that the last one is of type
  // ImageOverlayChunk.
  static_cast<ImageOverlayChunk*>(messages.back().get())->set_eof(true);

  // Since different clients may start playing the video at different times, we need to send the
  // video header with every chunk, instead of just once at the beginning.
  auto header = std::make_unique<VideoHeader>();
  header->set_height(video_header_.height);
  header->set_width(video_header_.width);
  header->set_frame_rate(video_header_.frame_rate);
  messages.push_back(std::move(header));

  auto cb = control_ctx->GetVideoWithOverlaysCallback();
  if (!!cb) {
    GML_RETURN_IF_ERROR(cb(messages));
  }

  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Add(new bool(true), cc->InputTimestamp());
  }

  return Status::OK();
}

REGISTER_CALCULATOR(OverlayedFFmpegVideoSinkCalculator);

}  // namespace gml::gem::calculators::ffmpeg
