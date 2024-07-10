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

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto gml_meter = metrics_system.GetMeterProvider()->GetMeter("gml");
  fps_gauge_ = gml_meter->CreateDoubleGauge("gml.gem.camera.fps");

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

  fps_gauge_->Record(options_.target_frame_rate(),
                     {{"camera_id", options_.device_uuid()}, {"camera", "argus"}}, {});

  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  argus_cam_->Stop();
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ArgusCamSourceCalculator);

}  // namespace gml::gem::calculators::argus
