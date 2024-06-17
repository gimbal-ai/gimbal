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

#include "src/gem/calculators/core/bytetrack_calculator.h"

#include <ByteTrack/BYTETracker.h>

#include <memory>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/detection.pb.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kDetectionVectorTag = "DETECTIONS";

int LabelMapper::get_id(const std::string& label) {
  int label_id;
  auto it = label_to_id_.find(label);
  if (it != label_to_id_.end()) {
    label_id = it->second;
  } else {
    // Assign new ID based on map size.
    label_id = static_cast<int>(label_to_id_.size());
    label_to_id_[label] = label_id;
  }

  return label_id;
}

absl::Status ByteTrackCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  cc->Outputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Open(mediapipe::CalculatorContext*) {
  byte_tracker_ = std::make_unique<byte_track::BYTETracker>();
  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Process(mediapipe::CalculatorContext* cc) {
  std::vector<byte_track::Object> tracker_objects;
  const auto& detections = cc->Inputs().Tag(kDetectionVectorTag).Get<std::vector<Detection>>();

  // Convert to bytetracker format.
  for (const auto& detection : detections) {
    const byte_track::Rect rect(detection.bounding_box().xc(), detection.bounding_box().yc(),
                                detection.bounding_box().width(),
                                detection.bounding_box().height());
    std::string label = detection.label(0).label();
    float score = detection.label(0).score();

    int label_id = label_mapper_.get_id(label);

    tracker_objects.emplace_back(rect, label_id, score);
  }

  std::vector<byte_track::BYTETracker::STrackPtr> tracked_objs =
      byte_tracker_->update(tracker_objects);

  // Convert back to detections format.
  std::vector<Detection> tracked_detections;
  for (const auto& tracked_obj : tracked_objs) {
    auto rect = tracked_obj->getRect();

    Detection detection;
    detection.mutable_bounding_box()->set_xc(rect.x());
    detection.mutable_bounding_box()->set_yc(rect.y());
    detection.mutable_bounding_box()->set_width(rect.width());
    detection.mutable_bounding_box()->set_height(rect.height());

    auto label = detection.add_label();
    // TODO(oazizi): Modify bytetrack cpp to preserve the original label so it can be copied.
    // For now, use track id instead so it will be rendered.
    // Assumes that there is only one class (and thus the label is irrelevant).
    label->set_label(absl::Substitute("id: $0", tracked_obj->getTrackId()));
    label->set_score(tracked_obj->getScore());

    detection.mutable_track_id()->set_value(static_cast<int64_t>(tracked_obj->getTrackId()));

    tracked_detections.push_back(detection);
  }

  auto packet = mediapipe::MakePacket<std::vector<Detection>>(std::move(tracked_detections));
  packet = packet.At(cc->InputTimestamp());
  cc->Outputs().Tag(kDetectionVectorTag).AddPacket(std::move(packet));

  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Close(mediapipe::CalculatorContext*) { return absl::OkStatus(); }

REGISTER_CALCULATOR(ByteTrackCalculator);

}  // namespace gml::gem::calculators::core
