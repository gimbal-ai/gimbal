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

TEST(ClockLatencyMetricsSinkCalculatorTest, Basic) {
  constexpr char kGraph[] = R"pbtxt(
    calculator: "ClockLatencyMetricsSinkCalculator"
    input_stream: "duration"
    output_stream: "FINISHED:finished"
    node_options {
      [type.googleapis.com/gml.gem.calculators.core.optionspb.ClockLatencyMetricsSinkCalculatorOptions] {
        name: "detect";
      }
    }
  )pbtxt";

  testing::CalculatorTester tester(kGraph);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  metrics_system.Reset();

  tester.ForInput(0, absl::Milliseconds(10), mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<bool>("FINISHED", 0, mediapipe::Timestamp(0), true);

  auto check_results = [](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
    const auto& scope_metrics = resource_metrics.scope_metric_data_;
    ASSERT_EQ(1, scope_metrics.size());

    const auto& metric_data = scope_metrics[0].metric_data_;
    ASSERT_EQ(1, metric_data.size());

    const auto& name = metric_data[0].instrument_descriptor.name_;
    const auto& point_data = metric_data[0].point_data_attr_;

    ASSERT_EQ(point_data.size(), 1);
    ASSERT_EQ("gml_gem_detect_latency_seconds", name);
    EXPECT_THAT(
        point_data[0],
        MatchHistogram(ExpectedHist{
            {0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.075, 0.1, 0.15},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {},
        }));
  };
  auto results_cb =
      [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        check_results(resource_metrics);
        return true;
      };
  metrics_system.Reader()->Collect(results_cb);
}

TEST(ClockLatencyMetricsSinkCalculatorTest, MultipleInputs) {
  constexpr char kGraph[] = R"pbtxt(
      calculator: "ClockLatencyMetricsSinkCalculator"
      input_stream: "duration0"
      input_stream: "duration1"
      output_stream: "FINISHED:finished"
      node_options {
        [type.googleapis.com/gml.gem.calculators.core.optionspb.ClockLatencyMetricsSinkCalculatorOptions] {
          name: "detect";
        }
      }
    )pbtxt";

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  metrics_system.Reset();

  testing::CalculatorTester tester(kGraph);

  tester.ForInput(0, absl::Milliseconds(5), mediapipe::Timestamp(0))
      .ForInput(1, absl::Milliseconds(10), mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<bool>("FINISHED", 0, mediapipe::Timestamp(0), true);

  auto check_results = [](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
    const auto& scope_metrics = resource_metrics.scope_metric_data_;
    ASSERT_EQ(1, scope_metrics.size());

    const auto& metric_data = scope_metrics[0].metric_data_;
    ASSERT_EQ(1, metric_data.size());

    const auto& name = metric_data[0].instrument_descriptor.name_;
    const auto& point_data = metric_data[0].point_data_attr_;

    ASSERT_EQ(point_data.size(), 1);
    ASSERT_EQ("gml_gem_detect_latency_seconds", name);
    EXPECT_THAT(
        point_data[0],
        MatchHistogram(ExpectedHist{
            {0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.075, 0.1, 0.15},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {},
        }));
  };
  auto results_cb =
      [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        check_results(resource_metrics);
        return true;
      };
  metrics_system.Reader()->Collect(results_cb);
}

}  // namespace gml::gem::calculators::core
