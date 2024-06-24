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
#include "src/gem/calculators/core/optionspb/bytetrack_calculator_options.pb.h"

namespace gml::gem::calculators::core {
using ::gml::gem::calculators::core::optionspb::ByteTrackCalculatorOptions;
using ::gml::internal::api::core::v1::Detection;

constexpr std::string_view kDetectionVectorTag = "DETECTIONS";

int LabelMapper::get_id(const std::string& label) {
  int label_id;
  auto it = label_to_id_.find(label);
  if (it != label_to_id_.end()) {
    // Found existing id.
    label_id = it->second;
  } else {
    // Assign new ID based on map size.
    label_id = static_cast<int>(label_to_id_.size());
    label_to_id_[label] = label_id;
    id_to_label_[label_id] = label;
  }

  return label_id;
}

StatusOr<std::string> LabelMapper::get_label(int id) {
  auto it = id_to_label_.find(id);
  if (it != id_to_label_.end()) {
    return it->second;
  }
  return error::NotFound("Could not find label for id: $0", id);
}

absl::Status ByteTrackCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  cc->Outputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<ByteTrackCalculatorOptions>();

  // Get proto values, or use default value if not set.
  int32_t max_frames_lost = 30;
  if (options_.has_max_frames_lost()) {
    max_frames_lost = options_.max_frames_lost().value();
  }

  float track_thresh = 0.5;
  if (options_.has_track_thresh()) {
    track_thresh = options_.track_thresh().value();
  }

  float high_thresh = 0.6;
  if (options_.has_high_thresh()) {
    high_thresh = options_.high_thresh().value();
  }

  float match_thresh = 0.8;
  if (options_.has_match_thresh()) {
    match_thresh = options_.match_thresh().value();
  }

  // Bytetrack uses the following relationship.
  //   max_frames_lost = frame_rate / 30.0 * track_buffer
  // We set frame_rate and track_buffer to achieve desired max_frames_lost.
  int frame_rate = 30;
  int track_buffer = max_frames_lost;

  byte_tracker_ = std::make_unique<byte_track::BYTETracker>(frame_rate, track_buffer, track_thresh,
                                                            high_thresh, match_thresh);

  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Process(mediapipe::CalculatorContext* cc) {
  std::vector<byte_track::Object> tracker_objects;
  const auto& detections = cc->Inputs().Tag(kDetectionVectorTag).Get<std::vector<Detection>>();

  // Convert to bytetracker format.
  for (const auto& detection : detections) {
    const byte_track::Rect rect(
        detection.bounding_box().xc() - detection.bounding_box().width() / 2,
        detection.bounding_box().yc() - detection.bounding_box().height() / 2,
        detection.bounding_box().width(), detection.bounding_box().height());
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
    detection.mutable_bounding_box()->set_xc(rect.x() + rect.width() / 2);
    detection.mutable_bounding_box()->set_yc(rect.y() + rect.height() / 2);
    detection.mutable_bounding_box()->set_width(rect.width());
    detection.mutable_bounding_box()->set_height(rect.height());

    auto label = detection.add_label();
    auto label_str = label_mapper_.get_label(tracked_obj->getClass()).ValueOr("");
    label->set_label(absl::Substitute("$0 $1", label_str, tracked_obj->getTrackId()));
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
