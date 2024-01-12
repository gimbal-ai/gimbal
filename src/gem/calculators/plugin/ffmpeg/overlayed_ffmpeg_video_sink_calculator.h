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
 *    DETECTIONS std::vector<internal::api::corepb::v1::Detection> optional list of detection protos
 *      (currently required until we support segmenatation or other overlays.
 *
 *    AV_PACKETS std::vector<std::unique_ptr<AVPacketWrapper>> list of ffmpeg
 *      encoded packets.
 *
 *    IMAGE_HIST internal::api::core::v1::ImageHistogram
 *    IMAGE_QUALITY internal::api::core::v1::ImageQualityMetrics.
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
