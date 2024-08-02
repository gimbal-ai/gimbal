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

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/otel_utils.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;
using ::gml::internal::api::core::v1::TracksMetadata;

using ::opentelemetry::sdk::common::OwnedAttributeValue;

namespace otel_metrics = opentelemetry::sdk::metrics;

struct MetricValues {
  ExpectedGaugeOrCounter<int64_t> active_tracks;
  ExpectedGaugeOrCounter<int64_t> unique_track_ids_count;
  ExpectedGaugeOrCounter<int64_t> lost_tracks;
  // Optional because histogram is not emitted before tracks are removed.
  std::optional<ExpectedHist> expected_track_frames_histogram;
  std::optional<ExpectedHist> expected_track_lifetime_histogram;
};

struct TracksMetricsSinkTestStep {
  std::vector<std::string> detection_pbtxts;
  std::vector<int64_t> removed_track_ids;
  MetricValues expected_metrics;
  int64_t apply_times = 1;
};

struct TracksMetricsSinkTestCase {
  std::string calculator_pbtxt;
  std::vector<TracksMetricsSinkTestStep> steps;
};

class TracksMetricsSinkTest : public ::testing::TestWithParam<TracksMetricsSinkTestCase> {};

TEST_P(TracksMetricsSinkTest, RecordsMetricsCorrectly) {
  auto test_case_parent = GetParam();
  auto ts = mediapipe::Timestamp::Min();
  int64_t test_i = 0;
  testing::CalculatorTester tester(test_case_parent.calculator_pbtxt);
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  for (const auto& test_case : test_case_parent.steps) {
    SCOPED_TRACE(absl::StrCat("Test step ", test_i));
    test_i++;

    std::vector<Detection> detections(test_case.detection_pbtxts.size());
    for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
      ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                                &detections[i]));
    }
    TracksMetadata tracks_metadata;
    tracks_metadata.mutable_removed_track_ids()->Assign(test_case.removed_track_ids.begin(),
                                                        test_case.removed_track_ids.end());

    // Increment the ts for the next step.
    for (int64_t i = 0; i < test_case.apply_times; i++) {
      metrics_system.Reset();
      ts = mediapipe::Timestamp(
          ts.Value() +
          static_cast<int64_t>(std::round(mediapipe::Timestamp::kTimestampUnitsPerSecond * 0.3)));
      tester.ForInput("DETECTIONS", detections, ts)
          .ForInput("TRACKS_METADATA", tracks_metadata, ts)
          .Run()
          .ExpectOutput<bool>("FINISHED", ts, true);

      if (i != test_case.apply_times - 1) {
        metrics_system.Reader()->Collect(
            [](opentelemetry::sdk::metrics::ResourceMetrics&) { return true; });
      }
    }

    auto check_results = [&test_case](
                             opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
      const auto& scope_metrics = resource_metrics.scope_metric_data_;
      ASSERT_EQ(1, scope_metrics.size());

      const auto& metric_data = scope_metrics[0].metric_data_;
      // Ensure that the number of metrics matches the test cases. There are 3 metrics that will
      // always be output and 2 more for histograms which are only generated when track_ids are
      // removed.
      ASSERT_EQ(3 + test_case.expected_metrics.expected_track_frames_histogram.has_value() +
                    test_case.expected_metrics.expected_track_lifetime_histogram.has_value(),
                metric_data.size());

      for (const auto& metric_datum : metric_data) {
        const auto& name = metric_datum.instrument_descriptor.name_;
        const auto& point_data = metric_datum.point_data_attr_;
        ASSERT_EQ(1, point_data.size());

        // Check the metric value based on its type
        if (name == "gml_gem_pipe_tracks_active") {
          EXPECT_THAT(point_data[0], MatchGauge<int64_t>(test_case.expected_metrics.active_tracks));
        } else if (name == "gml_gem_pipe_tracks_lost") {
          EXPECT_THAT(point_data[0], MatchGauge<int64_t>(test_case.expected_metrics.lost_tracks));
        } else if (name == "gml_gem_pipe_tracks_unique_ids") {
          EXPECT_THAT(point_data[0],
                      MatchCounter<int64_t>(test_case.expected_metrics.unique_track_ids_count));
        } else if (name == "gml_gem_pipe_tracks_frame_count") {
          if (test_case.expected_metrics.expected_track_frames_histogram.has_value()) {
            EXPECT_THAT(
                point_data[0],
                MatchHistogram(test_case.expected_metrics.expected_track_frames_histogram.value()));
          } else {
            FAIL() << "Unexpected gml_gem_pipe_tracks_frame_count: result " << point_data[0];
          }
        } else if (name == "gml_gem_pipe_tracks_lifetime") {
          if (test_case.expected_metrics.expected_track_lifetime_histogram.has_value()) {
            EXPECT_THAT(point_data[0],
                        MatchHistogram(
                            test_case.expected_metrics.expected_track_lifetime_histogram.value()));
          } else {
            FAIL() << "Unexpected gml_gem_pipe_tracks_lifetime: result " << point_data[0];
          }
        } else {
          FAIL() << "Unexpected metric name: " << name;
        }
      }
    };

    auto results_cb =
        [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
          // Wrap this call because ASSERT_* macros return void.
          check_results(resource_metrics);
          return true;
        };
    metrics_system.Reader()->Collect(results_cb);
  }
}

INSTANTIATE_TEST_SUITE_P(
    TracksMetricsSinkTestSuite, TracksMetricsSinkTest,
    ::testing::Values(
        TracksMetricsSinkTestCase{
            R"pbtxt(
                calculator: "TracksMetricsSinkCalculator"
                input_stream: "DETECTIONS:detections"
                input_stream: "TRACKS_METADATA:tracks_metadata"
                output_stream: "FINISHED:finished"
            )pbtxt",
            {TracksMetricsSinkTestStep{
              .detection_pbtxts = {
                R"pbtxt(
track_id { value: 1 }
label { label: "person" score: 0.9 }
bounding_box { xc: 0.5 yc: 0.5 width: 0.1 height: 0.2 }
)pbtxt",
                                               R"pbtxt(
track_id { value: 2 }
label { label: "car" score: 0.8 }
bounding_box { xc: 0.3 yc: 0.4 width: 0.2 height: 0.1 }
)pbtxt",
                                               R"pbtxt(
track_id { value: 3 }
label { label: "bicycle" score: 0.85 }
bounding_box { xc: 0.7 yc: 0.6 width: 0.15 height: 0.25 }
)pbtxt",
                                           },
                                       .removed_track_ids = {},
                                       .expected_metrics =
                                           {
                                               .active_tracks = {3, {}},
                                               .unique_track_ids_count = {3, {}},
                                               .lost_tracks = {0, {}},
                                               .expected_track_frames_histogram = std::nullopt,
                                               .expected_track_lifetime_histogram = std::nullopt,
                                           }},
             TracksMetricsSinkTestStep{.detection_pbtxts =
                                           {
                                               R"pbtxt(
        track_id { value: 1 }
        label { label: "person" score: 0.9 }
        bounding_box { xc: 0.5 yc: 0.5 width: 0.1 height: 0.2 }
        )pbtxt",
                                               R"pbtxt(
        track_id { value: 3 }
        label { label: "bicycle" score: 0.85 }
        bounding_box { xc: 0.7 yc: 0.6 width: 0.15 height: 0.25 }
        )pbtxt",
                                           },
                                       .removed_track_ids = {},
                                       .expected_metrics =
                                           {
                                               .active_tracks = {2, {}},
                                               .unique_track_ids_count = {3, {}},
                                               .lost_tracks = {1, {}},
                                               .expected_track_frames_histogram = std::nullopt,
                                               .expected_track_lifetime_histogram = std::nullopt,
                                           },
                                       .apply_times = 5},
             TracksMetricsSinkTestStep{
                 .detection_pbtxts =
                     {
                         R"pbtxt(
        track_id { value: 1 }
        label { label: "person" score: 0.9 }
        bounding_box { xc: 0.5 yc: 0.5 width: 0.1 height: 0.2 }
        )pbtxt",
                     },
                 .removed_track_ids = {2, 3},
                 .expected_metrics =
                     {
                         .active_tracks = {1, {}},
                         .unique_track_ids_count = {3, {}},
                         .lost_tracks = {0, {}},
                         .expected_track_frames_histogram = {{metrics::kDefaultHistogramBounds,
                                                              {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0, 0, 0, 0},
                                                              {}}},
                         .expected_track_lifetime_histogram = {{kTrackLifetimeHistogramBounds,
                                                              {1, 0,0,  1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,
                                                               0, 0, 0, 0},
                                                              {}}},
                     },
             }

            }},
        TracksMetricsSinkTestCase{
            R"pbtxt(
calculator: "TracksMetricsSinkCalculator"
node_options {
  [type.googleapis.com/gml.gem.calculators.core.optionspb.TracksMetricsSinkCalculatorOptions] {
    metric_attributes {
      key: "pipeline_id"
      value: "test_pipeline"
    }
    metric_attributes {
      key: "device_id"
      value: "test_device"
    }
  }
}
input_stream: "DETECTIONS:detections"
input_stream: "TRACKS_METADATA:tracks_metadata"
output_stream: "FINISHED:finished"
)pbtxt",
            {
                TracksMetricsSinkTestStep{
                    .detection_pbtxts = {
                                              R"pbtxt(
track_id { value: 1 }
label { label: "person" score: 0.9 }
bounding_box { xc: 0.5 yc: 0.5 width: 0.1 height: 0.2 }
)pbtxt",
                                          },
                                          .removed_track_ids = {},
                                          .expected_metrics = {
                                              .active_tracks={1,
                                               {
                                                   {
                                                       "pipeline_id",
                                                       "test_pipeline",
                                                   },
                                                   {
                                                       "device_id",
                                                       "test_device",
                                                   },
                                               }},
                                              .unique_track_ids_count={1,
                                               {{
                                                   {
                                                       "pipeline_id",
                                                       "test_pipeline",
                                                   },
                                                   {
                                                       "device_id",
                                                       "test_device",
                                                   },
                                               }}},
                                              .lost_tracks={0,
                                               {
                                                   {
                                                       "pipeline_id",
                                                       "test_pipeline",
                                                   },
                                                   {
                                                       "device_id",
                                                       "test_device",
                                                   },
                                               }},
                                              .expected_track_frames_histogram = std::nullopt,
                                          }},
                TracksMetricsSinkTestStep{
                    .detection_pbtxts={},
                    .removed_track_ids={1},
                    .expected_metrics={
                        .active_tracks={0,
                                          {
                          {
                              "pipeline_id",
                              "test_pipeline",
                          },
                          {
                              "device_id",
                              "test_device",
                          },
                      }},
                     .unique_track_ids_count={1,
                      {
                          {
                              "pipeline_id",
                              "test_pipeline",
                          },
                          {
                              "device_id",
                              "test_device",
                          },
                      }},
                     .lost_tracks={0,
                      {
                          {
                              "pipeline_id",
                              "test_pipeline",
                          },
                          {
                              "device_id",
                              "test_device",
                          },
                      }},
                     .expected_track_frames_histogram{{metrics::kDefaultHistogramBounds,
                       {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {{
                            "pipeline_id",
                            "test_pipeline",
                        },
                        {
                            "device_id",
                            "test_device",
                        }}}},
                      .expected_track_lifetime_histogram = {{kTrackLifetimeHistogramBounds,
                       {1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0},
                       {{
                            "pipeline_id",
                            "test_pipeline",
                        },
                        {
                            "device_id",
                            "test_device",
                        }}}}
                        },
                },

            },
        }));

}  // namespace gml::gem::calculators::core
