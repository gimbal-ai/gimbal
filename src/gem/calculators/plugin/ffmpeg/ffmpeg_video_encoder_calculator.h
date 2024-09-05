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
  absl::Status SetupCodec(int height, int width);
};

}  // namespace gml::gem::calculators::ffmpeg
