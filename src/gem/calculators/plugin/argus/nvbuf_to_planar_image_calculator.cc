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

#include "src/gem/calculators/plugin/argus/nvbuf_to_planar_image_calculator.h"

#include "absl/status/status.h"

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml {
namespace gem {
namespace calculators {
namespace argus {

using ::gml::gem::devices::argus::NvBufSurfaceWrapper;
using ::gml::gem::exec::core::ImageFormat;
using ::gml::gem::exec::core::PlanarImage;
using ::gml::gem::exec::core::PlanarImageFor;

absl::Status NvBufSurfToPlanarImageCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<NvBufSurfaceWrapper>();
  cc->Outputs().Index(0).Set<mediapipe::ImageFrame>();
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

}  // namespace argus
}  // namespace calculators
}  // namespace gem
}  // namespace gml
