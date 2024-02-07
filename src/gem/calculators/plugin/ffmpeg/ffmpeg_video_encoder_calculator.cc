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
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/video_stream_header.h>

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/ffmpeg/ffmpeg_video_encoder_calculator.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml::gem::calculators::ffmpeg {

using AVPacketWrappers = std::vector<std::unique_ptr<AVPacketWrapper>>;

constexpr std::string_view kPlanarImageTag = "PLANAR_IMAGE";
constexpr std::string_view kAVPacketsTag = "AV_PACKETS";
constexpr std::string_view kVideoHeaderTag = "VIDEO_HEADER";

// TODO(james): move these to options.
constexpr char kCodecName[] = "libopenh264\0";
constexpr int64_t kTargetKiloBitrate = 500;
constexpr int64_t kGOPSize = 30;

absl::Status FFmpegVideoEncoderCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kPlanarImageTag).Set<std::unique_ptr<exec::core::PlanarImage>>();
  cc->Inputs().Tag(kVideoHeaderTag).Set<mediapipe::VideoHeader>();
  cc->Outputs().Tag(kAVPacketsTag).Set<AVPacketWrappers>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status FFmpegVideoEncoderCalculator::Open(mediapipe::CalculatorContext*) {
  const AVCodec* codec = avcodec_find_encoder_by_name(kCodecName);
  if (!codec) {
    return {absl::StatusCode::kInvalidArgument, "Codec not found"};
  }
  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    return {absl::StatusCode::kInternal, "Failed to allocate codec context"};
  }
  av_packet_ = AVPacketWrapper::Create();
  if (!av_packet_->packet()) {
    return {absl::StatusCode::kInternal, "Failed to allocate AVPacket"};
  }
  frame_ = av_frame_alloc();

  return absl::OkStatus();
}

absl::Status FFmpegVideoEncoderCalculator::SetupCodec(int height, int width, int frame_rate) {
  codec_ctx_->bit_rate = kTargetKiloBitrate * 1000;
  codec_ctx_->width = width;
  codec_ctx_->height = height;

  codec_ctx_->time_base = AVRational{1, frame_rate};
  codec_ctx_->framerate = AVRational{frame_rate, 1};

  // The openh264 encoder expects YUV420P format.
  codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;

  codec_ctx_->gop_size = kGOPSize;

  av_opt_set_int(codec_ctx_->priv_data, "b", kTargetKiloBitrate * 1000, 0);
  av_opt_set(codec_ctx_->priv_data, "preset", "constrained_baseline", 0);

  auto ret = avcodec_open2(codec_ctx_, codec_ctx_->codec, nullptr);
  if (ret < 0) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::Substitute("Failed avcodec_open2: $0", av_err2str(ret)));
  }

  frame_->format = codec_ctx_->pix_fmt;
  frame_->width = codec_ctx_->width;
  frame_->height = codec_ctx_->height;

  ret = av_frame_get_buffer(frame_, 0);
  if (ret < 0) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::Substitute("Failed to allocate frame buffer: $0", av_err2str(ret)));
  }

  codec_setup_ = true;

  return absl::OkStatus();
}

absl::Status FFmpegVideoEncoderCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (cc->InputTimestamp() == mediapipe::Timestamp::PreStream()) {
    const auto& video_header = cc->Inputs().Tag(kVideoHeaderTag).Get<mediapipe::VideoHeader>();
    return SetupCodec(video_header.height, video_header.width,
                      static_cast<int>(video_header.frame_rate));
  }

  if (!codec_setup_) {
    return {absl::StatusCode::kInternal, "No video header provided"};
  }

  const auto& planar_image =
      cc->Inputs().Tag(kPlanarImageTag).Get<std::unique_ptr<exec::core::PlanarImage>>();

  auto planes = planar_image->Planes();

  if (planes.size() != 3) {
    return {absl::StatusCode::kInvalidArgument,
            absl::Substitute("Expected YUV PlanarImage in 3-plane format, received $0 planes.",
                             planes.size())};
  }

  for (size_t i = 0; i < planes.size(); ++i) {
    std::memcpy(frame_->data[i], planes[i].data, planes[i].bytes);
    frame_->linesize[i] = planes[i].row_stride;
  }

  frame_->pts = cc->InputTimestamp().Value();

  auto ret = avcodec_send_frame(codec_ctx_, frame_);
  if (ret < 0) {
    return {absl::StatusCode::kInternal, "Error sending frame to ffmpeg for encoding"};
  }

  std::vector<std::unique_ptr<AVPacketWrapper>> av_packets;
  while (ret >= 0) {
    auto* mut_av_packet = av_packet_->mutable_packet();
    ret = avcodec_receive_packet(codec_ctx_, mut_av_packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      return {absl::StatusCode::kInternal, "Error during encoding"};
    }
    auto out_av_packet = AVPacketWrapper::CreateRef(mut_av_packet);
    av_packets.emplace_back(std::move(out_av_packet));

    // Unref the packet locally, the out_av_packet ref will prevent the packet's data from being
    // freed.
    av_packet_unref(mut_av_packet);
  }

  auto mp_packet = mediapipe::MakePacket<AVPacketWrappers>(std::move(av_packets));
  mp_packet = mp_packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kAVPacketsTag).AddPacket(std::move(mp_packet));

  return absl::OkStatus();
}

absl::Status FFmpegVideoEncoderCalculator::Close(mediapipe::CalculatorContext*) {
  avcodec_free_context(&codec_ctx_);
  av_frame_free(&frame_);

  return absl::OkStatus();
}

REGISTER_CALCULATOR(FFmpegVideoEncoderCalculator);

}  // namespace gml::gem::calculators::ffmpeg
