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

#include "src/gem/calculators/core/clock_latency_metrics_sink_calculator.h"

#include <iostream>

#include "gmock/gmock.h"
#include <absl/time/time.h>

#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/otel_utils.h"

namespace gml::gem::calculators::core {

const std::vector<double> kLatencyBucketBounds = {0,     0.005, 0.01,  0.015, 0.02,  0.025, 0.03,
                                                  0.035, 0.04,  0.045, 0.05,  0.075, 0.1,   0.15};

constexpr std::string_view kClockLatencyMetricsSinkNode = R"pbtxt(
calculator: "ClockLatencyMetricsSinkCalculator"
input_stream: "duration"
output_stream: "FINISHED:finished"
node_options {
  [type.googleapis.com/gml.gem.calculators.core.optionspb.ClockLatencyMetricsSinkCalculatorOptions] {
    name: "detect"
  }
}
)pbtxt";

struct ClockLatencyMetricsSinkTestCase {
  std::string_view config;
  std::vector<absl::Duration> input_durations;
  absl::flat_hash_map<std::string, ExpectedMetric> expected_metrics;
};

class ClockLatencyMetricsSinkTest
    : public ::testing::TestWithParam<ClockLatencyMetricsSinkTestCase> {};

TEST_P(ClockLatencyMetricsSinkTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  std::string config(test_case.config);
  testing::CalculatorTester tester(config);
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto ts = mediapipe::Timestamp::Min();

  for (int i = 0; i < static_cast<int>(test_case.input_durations.size()); ++i) {
    tester.ForInput(i, test_case.input_durations[i], ts);
  }
  tester.Run();

  CheckMetrics(metrics_system, test_case.expected_metrics);
  metrics_system.Reset();
}

INSTANTIATE_TEST_SUITE_P(
    ClockLatencyMetricsSinkTestSuite, ClockLatencyMetricsSinkTest,
    ::testing::Values(
        ClockLatencyMetricsSinkTestCase{
            .config = kClockLatencyMetricsSinkNode,
            .input_durations = {absl::Milliseconds(10)},
            .expected_metrics =
                {
                    {"gml_gem_detect_latency_seconds",
                     ExpectedHist{
                         .bucket_bounds = kLatencyBucketBounds,
                         .bucket_counts = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                     }},
                },
        },
        ClockLatencyMetricsSinkTestCase{
            .config = R"pbtxt(
calculator: "ClockLatencyMetricsSinkCalculator"
input_stream: "duration0"
input_stream: "duration1"
output_stream: "FINISHED:finished"
node_options {
  [type.googleapis.com/gml.gem.calculators.core.optionspb.ClockLatencyMetricsSinkCalculatorOptions] {
    name: "detect"
  }
}
)pbtxt",
            .input_durations = {absl::Milliseconds(5), absl::Milliseconds(10)},
            .expected_metrics =
                {
                    {"gml_gem_detect_latency_seconds",
                     ExpectedHist{
                         .bucket_bounds = kLatencyBucketBounds,
                         .bucket_counts = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                     }},
                },
        },
        ClockLatencyMetricsSinkTestCase{
            .config = kClockLatencyMetricsSinkNode,
            .input_durations = {absl::Milliseconds(-5)},
            .expected_metrics =
                {
                    {"gml_gem_detect_latency_seconds",
                     ExpectedHist{
                         .bucket_bounds = kLatencyBucketBounds,
                         .bucket_counts = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                     }},
                },
        },
        // No input
        ClockLatencyMetricsSinkTestCase{
            .config = kClockLatencyMetricsSinkNode,
            .input_durations = {},
            .expected_metrics = {},
        }));

}  // namespace gml::gem::calculators::core
