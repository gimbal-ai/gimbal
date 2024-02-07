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
#pragma once

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/video_stream_header.h>

#include "src/gem/calculators/plugin/ffmpeg/av_packet_wrapper.h"

namespace gml::gem::calculators::ffmpeg {

/**
 * FFMPEGVideoEncoderCalculator Graph API:
 *  Inputs:
 *    PLANAR_IMAGE std::unique_ptr<exec::core::PlanarImage>
 *    VIDEO_HEADER mediapipe::VideoHeader (optional)
 *
 *  Outputs:
 *    AV_PACKETS std::vector<std::unique_ptr<AVPacketWrapper>>
 *
 */
class FFmpegVideoEncoderCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  AVCodecContext* codec_ctx_;
  AVFrame* frame_;
  std::unique_ptr<AVPacketWrapper> av_packet_;

  bool codec_setup_ = false;
  absl::Status SetupCodec(int height, int width, int frame_rate);
};

}  // namespace gml::gem::calculators::ffmpeg
