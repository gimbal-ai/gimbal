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

#include "src/gem/calculators/plugin/argus/nvbuf_to_planar_image_calculator.h"

#include <absl/status/status.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml::gem::calculators::argus {

using ::gml::gem::devices::argus::NvBufSurfaceWrapper;
using ::gml::gem::exec::core::ImageFormat;
using ::gml::gem::exec::core::PlanarImage;
using ::gml::gem::exec::core::PlanarImageFor;

absl::Status NvBufSurfToPlanarImageCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<NvBufSurfaceWrapper>();
  cc->Outputs().Index(0).Set<mediapipe::ImageFrame>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status NvBufSurfToPlanarImageCalculator::Open(mediapipe::CalculatorContext* /* cc */) {
  return absl::OkStatus();
}

absl::Status NvBufSurfToPlanarImageCalculator::Process(mediapipe::CalculatorContext* cc) {
  mediapipe::Packet& in_packet = cc->Inputs().Index(0).Value();

  absl::StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> nvbuf_surf_s =
      in_packet.Consume<NvBufSurfaceWrapper>();
  if (!nvbuf_surf_s.ok()) {
    return nvbuf_surf_s.status();
  }

  std::unique_ptr<NvBufSurfaceWrapper> nvbuf_surf = std::move(nvbuf_surf_s.value());

  if (!nvbuf_surf->IsMapped()) {
    return absl::InternalError("The provided nvbuf is not mapped to CPU memory.");
  }

  GML_ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PlanarImage> planar_image,
      PlanarImageFor<NvBufSurfaceWrapper>::Create(std::move(nvbuf_surf), ImageFormat::YUV_I420));

  cc->Outputs().Index(0).Add(planar_image.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status NvBufSurfToPlanarImageCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(NvBufSurfToPlanarImageCalculator);

}  // namespace gml::gem::calculators::argus
