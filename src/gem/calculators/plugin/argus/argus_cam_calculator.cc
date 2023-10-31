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

#include "absl/status/status.h"

#include <mediapipe/framework/calculator_framework.h>
#include "src/gem/devices/camera/argus/argus_cam.h"

namespace gml {
namespace gem {
namespace calculators {
namespace argus {

using ::gml::gem::devices::argus::NvBufSurfaceWrapper;

absl::Status ArgusCamSourceCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Outputs().Index(0).Set<std::shared_ptr<NvBufSurfaceWrapper>>();
  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Open(mediapipe::CalculatorContext* /* cc */) {
  GML_ABSL_RETURN_IF_ERROR(argus_cam_.Init());
  timestamp_ = 0;
  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Process(mediapipe::CalculatorContext* cc) {
  absl::Status s;

  GML_ABSL_ASSIGN_OR_RETURN(std::unique_ptr<NvBufSurfaceWrapper> buf, argus_cam_.ConsumeFrame());

  // Convert to shared_ptr to give downstream mediapipe calculators flexibility,
  // specifically for use by the PlanarImage interface.
  std::shared_ptr<NvBufSurfaceWrapper> buf_shared = std::move(buf);

  auto packet = mediapipe::MakePacket<std::shared_ptr<NvBufSurfaceWrapper>>(std::move(buf_shared));
  packet = packet.At(mediapipe::Timestamp(timestamp_));
  cc->Outputs().Index(0).AddPacket(std::move(packet));

  timestamp_++;

  return absl::OkStatus();
}

absl::Status ArgusCamSourceCalculator::Close(mediapipe::CalculatorContext* /* cc */) {
  argus_cam_.Stop();
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ArgusCamSourceCalculator);

}  // namespace argus
}  // namespace calculators
}  // namespace gem
}  // namespace gml
