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
constexpr std::string_view kRemovedTrackIdsTag = "REMOVED_TRACK_IDS";

int LabelMapper::get_id(const std::string& label) {
  static int next_label_id = 0;

  StatusOr<int> res = label_id_bimap_.KeyToValue(label);
  if (res.ok()) {
    return res.ValueOrDie();
  }

  int label_id = next_label_id;
  next_label_id++;

  auto status = label_id_bimap_.Insert(label, label_id);
  if (!status.ok()) {
    LOG(ERROR) << absl::Substitute("Failed to insert label: $0 with id: $1", label, label_id);
    return -1;
  }
  return label_id;
}

StatusOr<std::string> LabelMapper::get_label(int id) { return label_id_bimap_.ValueToKey(id); }

absl::Status ByteTrackCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  cc->Outputs().Tag(kDetectionVectorTag).Set<std::vector<Detection>>();
  if (cc->Outputs().HasTag(kRemovedTrackIdsTag)) {
    cc->Outputs().Tag(kRemovedTrackIdsTag).Set<std::vector<int64_t>>();
  }
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

  const auto stracks = byte_tracker_->update(tracker_objects);
  const std::vector<byte_track::BYTETracker::STrackPtr> tracked_objs = stracks.active_stracks;

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

  if (cc->Outputs().HasTag(kRemovedTrackIdsTag)) {
    std::vector<int64_t> removed_track_ids;
    removed_track_ids.reserve(stracks.removed_stracks.size());
    for (const auto& strack : stracks.removed_stracks) {
      removed_track_ids.push_back(static_cast<int64_t>(strack->getTrackId()));
    }
    auto packet = mediapipe::MakePacket<std::vector<int64_t>>(std::move(removed_track_ids));
    packet = packet.At(cc->InputTimestamp());
    cc->Outputs().Tag(kRemovedTrackIdsTag).AddPacket(std::move(packet));
  }

  return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Close(mediapipe::CalculatorContext*) { return absl::OkStatus(); }

REGISTER_CALCULATOR(ByteTrackCalculator);

}  // namespace gml::gem::calculators::core
