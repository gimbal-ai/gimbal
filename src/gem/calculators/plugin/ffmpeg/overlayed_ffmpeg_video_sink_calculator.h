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

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/video_stream_header.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::calculators::ffmpeg {

/**
 *  OverlayedFFmpegVideoSinkCalculator Graph API:
 *
 *  Inputs:
 *    DETECTIONS std::vector<internal::api::corepb::v1::Detection> optional list of detection
 * protos.
 *
 *    AV_PACKETS std::vector<std::unique_ptr<AVPacketWrapper>> list of ffmpeg
 *      encoded packets.
 *
 *    IMAGE_HIST internal::api::core::v1::ImageHistogram
 *    IMAGE_QUALITY internal::api::core::v1::ImageQualityMetrics.
 *
 *    SEGMENTATION internal::api::corepb::v1::Segmentation optional proto with segmentation masks.
 *
 *  Outputs:
 *    This is a sink node so there are no data mediapipe outputs. Instead the node outputs proto
 *      data to the GEM controller through the ControlExecutionContext. However, the node does
 *      output a FINISHED packet, that has the same timestamp as the inputs, that can be used as a
 *      back edge for flow limiting.
 *
 *    FINISHED bool (always true)
 */
class OverlayedFFmpegVideoSinkCalculator : public core::ControlExecutionContextCalculator {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     exec::core::ControlExecutionContext* control_ctx) override;

 private:
  mediapipe::VideoHeader video_header_;
};

}  // namespace gml::gem::calculators::ffmpeg
