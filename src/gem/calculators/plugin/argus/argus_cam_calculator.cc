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

#include "src/gem/calculators/plugin/argus/argus_cam_calculator.h"

#include <absl/status/status.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/video_stream_header.h>

#include "src/gem/devices/camera/argus/argus_cam.h"
#include "src/gem/devices/camera/argus/argus_manager.h"

namespace gml::gem::calculators::argus {

constexpr std::string_view kVideoPrestreamTag = "VIDEO_PRESTREAM";

using ::gml::gem::calculators::argus::optionspb::ArgusCamSourceCalculatorOptions;
using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

absl::Status ArgusCamSourceCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Outputs().Index(0).Set<std::shared_ptr<NvBufSurfaceWrapper>>();
  if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
    cc->Outputs().Tag(kVideoPrestreamTag).Set<mediapipe::VideoHeader>();
  }
  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<ArgusCamSourceCalculatorOptions>();
  auto& argus_manager = devices::argus::ArgusManager::GetInstance();
  GML_ABSL_ASSIGN_OR_RETURN(
      argus_cam_, argus_manager.GetCamera(options_.device_uuid(), options_.target_frame_rate()));
  GML_ABSL_RETURN_IF_ERROR(argus_cam_->Init());

  // grab one frame to get metadata.
  GML_ABSL_ASSIGN_OR_RETURN(std::unique_ptr<NvBufSurfaceWrapper> buf, argus_cam_->ConsumeFrame());
  GML_ABSL_RETURN_IF_ERROR(buf->MapForCpu());

  auto header = std::make_unique<mediapipe::VideoHeader>();
  header->format = mediapipe::ImageFormat::FORMAT_SRGB;
  header->width = buf->surface().width;
  header->height = buf->surface().height;
  header->frame_rate = static_cast<double>(options_.target_frame_rate());

  if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
    cc->Outputs().Tag(kVideoPrestreamTag).Add(header.release(), mediapipe::Timestamp::PreStream());
    cc->Outputs().Tag(kVideoPrestreamTag).Close();
  }

  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Process(mediapipe::CalculatorContext* cc) {
  absl::Status s;

  GML_ABSL_ASSIGN_OR_RETURN(std::unique_ptr<NvBufSurfaceWrapper> buf, argus_cam_->ConsumeFrame());
  auto timestamp_ns_ = argus_cam_->GetLastCaptureNS();
  GML_ABSL_RETURN_IF_ERROR(buf->MapForCpu());
  // Convert to shared_ptr to give downstream mediapipe calculators flexibility,
  // specifically for use by the PlanarImage interface.
  std::shared_ptr<NvBufSurfaceWrapper> buf_shared = std::move(buf);

  auto packet = mediapipe::MakePacket<std::shared_ptr<NvBufSurfaceWrapper>>(std::move(buf_shared));
  packet = packet.At(mediapipe::Timestamp(timestamp_ns_ / 1000));
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  argus_cam_->Stop();
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ArgusCamSourceCalculator);

}  // namespace gml::gem::calculators::argus
