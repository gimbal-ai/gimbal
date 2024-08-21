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

#pragma once

#include <mediapipe/framework/calculator_framework.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/status.h"

namespace gml::gem::calculators::core {

// TODO(philkuz) Consider using an exponential histogram. Will need testing in victoria db and the
// metrics dashboard.
static const std::vector<double> kTrackLifetimeHistogramBounds = {
    0, 0.3, 1, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};

/**
 *  TrackingMetricsSinkCalculator Graph API:
 *
 *  Inputs:
 *    std::vector<internal::api::corepb::v1::Detection> list of detection protos that should contain
 *        the tracking id.
 *    std::vector<internal::api::corepb::v1::TracksMetadata> The metadata about tracks currently
 *        held in the tracker. In the future, we can exploring making this optional as it currently
 *        only provides info about removed ids, which we can approximate with a timeout on this
 *        metric node.
 *
 *  No real outputs, outputs stats to opentelemetry. Optional bool output to signal when processing
 *    is finished.
 */
class TracksMetricsSinkCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);

  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext*) override { return absl::OkStatus(); }

 private:
  struct TrackInfo {
    int64_t track_id;
    int64_t frame_count;
    mediapipe::Timestamp first_timestamp;
    mediapipe::Timestamp latest_timestamp;
  };
  opentelemetry::metrics::Gauge<uint64_t>* active_tracks_gauge_;
  opentelemetry::metrics::Counter<uint64_t>* unique_track_ids_counter_;
  opentelemetry::metrics::Histogram<uint64_t>* track_frame_histogram_;
  opentelemetry::metrics::Histogram<double>* track_lifetime_histogram_;
  absl::flat_hash_map<int64_t, TrackInfo> track_id_to_info_;
};

}  // namespace gml::gem::calculators::core
