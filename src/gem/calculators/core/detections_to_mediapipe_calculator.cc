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

#include "src/gem/calculators/core/detections_to_mediapipe_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/detection.pb.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kDetectionVectorTag = "DETECTIONS";
constexpr std::string_view kSingleDetectionTag = "DETECTION";

namespace {
void GMLToMediapipe(const Detection& detection, mediapipe::Detection* mp_detection) {
  auto xc = detection.bounding_box().xc();
  auto yc = detection.bounding_box().yc();
  auto width = detection.bounding_box().width();
  auto height = detection.bounding_box().height();

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
}  // namespace

absl::Status DetectionsToMediapipeCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kDetectionVectorTag)) {
    cc->Inputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
    cc->Outputs().Tag(kDetectionVectorTag).Set<std::vector<mediapipe::Detection>>();
  } else if (cc->Inputs().HasTag(kSingleDetectionTag)) {
    cc->Inputs().Tag(kSingleDetectionTag).Set<Detection>();
    cc->Outputs().Tag(kSingleDetectionTag).Set<mediapipe::Detection>();
  } else {
    return {absl::StatusCode::kInvalidArgument, "Must specify DETECTIONS or DETECTION tag"};
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Open(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kDetectionVectorTag)) {
    const auto& detections = cc->Inputs().Tag(kDetectionVectorTag).Get<std::vector<Detection>>();
    std::vector<mediapipe::Detection> mp_detections(detections.size());
    for (const auto& [i, detection] : Enumerate(detections)) {
      GMLToMediapipe(detection, &mp_detections[i]);
    }
    auto packet =
        mediapipe::MakePacket<std::vector<mediapipe::Detection>>(std::move(mp_detections));
    packet = packet.At(cc->InputTimestamp());
    cc->Outputs().Tag(kDetectionVectorTag).AddPacket(std::move(packet));
  } else {
    const auto& detection = cc->Inputs().Tag(kSingleDetectionTag).Get<Detection>();
    auto mp_detection = std::make_unique<mediapipe::Detection>();
    GMLToMediapipe(detection, mp_detection.get());
    cc->Outputs().Tag(kSingleDetectionTag).Add(mp_detection.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

absl::Status DetectionsToMediapipeCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionsToMediapipeCalculator);

}  // namespace gml::gem::calculators::core
