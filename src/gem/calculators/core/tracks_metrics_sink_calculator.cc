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

#include "src/gem/calculators/core/tracks_metrics_sink_calculator.h"

#include <absl/strings/str_cat.h>
#include <opentelemetry/metrics/provider.h>

#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/tracks_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kDetectionsTag = "DETECTIONS";
constexpr std::string_view kTracksMetadataTag = "TRACKS_METADATA";
constexpr std::string_view kFinishedTag = "FINISHED";

absl::Status TracksMetricsSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionsTag).Set<std::vector<::gml::internal::api::core::v1::Detection>>();
  cc->Inputs().Tag(kTracksMetadataTag).Set<::gml::internal::api::core::v1::TracksMetadata>();
  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Set<bool>();
  }
  cc->SetTimestampOffset(0);

  return absl::OkStatus();
}

absl::Status TracksMetricsSinkCalculator::Open(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  active_tracks_gauge_ = metrics_system.CreateGauge<uint64_t>(
      "gml_gem_active_tracks", "Count of tracks that are detected in the current frame");
  lost_tracks_gauge_ = metrics_system.CreateGauge<uint64_t>(
      "gml_gem_lost_tracks",
      "Count of tracks that have not been removed from the tracker and are not currently detected "
      "in the current frame");
  unique_track_ids_counter_ =
      metrics_system.CreateCounter("gml_gem_unique_track_ids_count", "Count of unique track IDs");

  track_frame_histogram_ = metrics_system.CreateHistogramWithBounds<uint64_t>(
      "gml_gem_track_frames",
      "Distribution of frame counts for tracked objects when they are removed",
      metrics::kDefaultHistogramBounds);
  track_lifetime_histogram_ = metrics_system.CreateHistogramWithBounds<double>(
      "gml_gem_track_lifetime", "Distribution of track lifetimes in milliseconds",
      kTrackLifetimeHistogramBounds);

  return absl::OkStatus();
}

absl::Status TracksMetricsSinkCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::TracksMetricsSinkCalculatorOptions>();
  const auto& detections = cc->Inputs()
                               .Tag(kDetectionsTag)
                               .Get<std::vector<::gml::internal::api::core::v1::Detection>>();
  const auto& tracks_metadata =
      cc->Inputs().Tag(kTracksMetadataTag).Get<::gml::internal::api::core::v1::TracksMetadata>();

  uint64_t new_track_id_count = 0;
  for (const auto& detection : detections) {
    if (!detection.has_track_id()) {
      continue;
    }
    if (!track_id_to_info_.contains(detection.track_id().value())) {
      new_track_id_count++;
      track_id_to_info_[detection.track_id().value()] =
          TrackInfo{.track_id = detection.track_id().value(),
                    .frame_count = 1,
                    .first_timestamp = cc->InputTimestamp(),
                    .latest_timestamp = cc->InputTimestamp()};
    } else {
      auto& track_info = track_id_to_info_[detection.track_id().value()];
      track_info.latest_timestamp = cc->InputTimestamp();
      track_info.frame_count++;
    }
  }

  // TODO(philkuz) do we want to record the class of the detection as an attribute?
  // pros: we can get track stats about each class, cons: the class may change
  // during the track lifetime.
  std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                     options.metric_attributes().end());
  for (const auto& track_id : tracks_metadata.removed_track_ids()) {
    if (!track_id_to_info_.contains(track_id)) {
      continue;
    }
    auto& track_info = track_id_to_info_[track_id];
    track_frame_histogram_->Record(track_info.frame_count, attrs);
    track_lifetime_histogram_->Record(
        track_info.latest_timestamp.Seconds() - track_info.first_timestamp.Seconds(), attrs);
    track_id_to_info_.erase(track_id);
  }

  lost_tracks_gauge_->Record(track_id_to_info_.size() - detections.size(), attrs);
  active_tracks_gauge_->Record(detections.size(), attrs);

  unique_track_ids_counter_->Add(new_track_id_count, attrs);

  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs()
        .Tag(kFinishedTag)
        .AddPacket(mediapipe::MakePacket<bool>(true).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

REGISTER_CALCULATOR(TracksMetricsSinkCalculator);
}  // namespace gml::gem::calculators::core
