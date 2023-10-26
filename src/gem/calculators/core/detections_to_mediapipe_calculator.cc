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
#include "src/shared/modelout/v1/detection.pb.h"

namespace gml {
namespace gem {
namespace calculators {
namespace core {
using ::gml::shared::modelout::v1::ImageDetections;

absl::Status DetectionsToMediapipeCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<ImageDetections>();
  cc->Outputs().Index(0).Set<mediapipe::DetectionList>();
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto image_detections = cc->Inputs().Index(0).Get<ImageDetections>();
  mediapipe::DetectionList output_detections;

  for (const auto& detection : image_detections.detection()) {
    auto xc = detection.bounding_box().xc();
    auto yc = detection.bounding_box().yc();
    auto width = detection.bounding_box().width();
    auto height = detection.bounding_box().height();

    auto* mp_detection = output_detections.add_detection();
    mp_detection->add_label(detection.label(0).label());
    mp_detection->add_score(detection.label(0).score());
    auto* location_data = mp_detection->mutable_location_data();
    location_data->set_format(mediapipe::LocationData::RELATIVE_BOUNDING_BOX);
    auto* bounding_box = location_data->mutable_relative_bounding_box();
    bounding_box->set_xmin(xc - (width / 2));
    bounding_box->set_ymin(yc - (height / 2));
    bounding_box->set_width(width);
    bounding_box->set_height(height);
  }
  auto packet = mediapipe::MakePacket<mediapipe::DetectionList>(std::move(output_detections));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Index(0).AddPacket(std::move(packet));
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionsToMediapipeCalculator);

}  // namespace core
}  // namespace calculators
}  // namespace gem
}  // namespace gml
