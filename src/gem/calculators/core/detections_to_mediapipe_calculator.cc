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

#include "src/gem/calculators/core/detections_to_mediapipe_calculator.h"
#include <mediapipe/framework/calculator_registry.h>
#include "mediapipe/framework/formats/detection.pb.h"
#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

absl::Status DetectionsToMediapipeCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::vector<Detection>>();
  cc->Outputs().Index(0).Set<std::vector<mediapipe::Detection>>();
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& detections = cc->Inputs().Index(0).Get<std::vector<Detection>>();
  std::vector<mediapipe::Detection> mp_detections(detections.size());

  for (const auto& [i, detection] : Enumerate(detections)) {
    auto xc = detection.bounding_box().xc();
    auto yc = detection.bounding_box().yc();
    auto width = detection.bounding_box().width();
    auto height = detection.bounding_box().height();

    auto* mp_detection = &mp_detections[i];
    mp_detection->add_label(detection.label(0).label());
    mp_detection->add_score(detection.label(0).score());
    auto* location_data = mp_detection->mutable_location_data();
    location_data->set_format(mediapipe::LocationData::LOCATION_FORMAT_RELATIVE_BOUNDING_BOX);
    auto* bounding_box = location_data->mutable_relative_bounding_box();
    bounding_box->set_xmin(xc - (width / 2));
    bounding_box->set_ymin(yc - (height / 2));
    bounding_box->set_width(width);
    bounding_box->set_height(height);
  }
  auto packet = mediapipe::MakePacket<std::vector<mediapipe::Detection>>(std::move(mp_detections));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Index(0).AddPacket(std::move(packet));
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionsToMediapipeCalculator);

}  // namespace gml::gem::calculators::core
