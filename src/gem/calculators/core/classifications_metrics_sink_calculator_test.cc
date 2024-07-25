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

#include "src/gem/calculators/core/classifications_metrics_sink_calculator.h"

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Label;

// TODO(oazizi): Refactor structs that are shared with detections_summary_calculator_test.cc.
struct ExpectedHist {
  std::vector<double> bucket_bounds;
  std::vector<uint64_t> bucket_counts;
  absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue> attributes;
};

auto MatchPointData(const ExpectedHist& expected) {
  using ::testing::AllOf;
  using ::testing::ElementsAreArray;
  using ::testing::Field;
  using ::testing::UnorderedElementsAreArray;
  using ::testing::VariantWith;

  namespace otel_metrics = opentelemetry::sdk::metrics;

  return AllOf(
      Field(&otel_metrics::PointDataAttributes::point_data,
            VariantWith<otel_metrics::HistogramPointData>(
                AllOf(Field(&otel_metrics::HistogramPointData::boundaries_,
                            ElementsAreArray(expected.bucket_bounds)),
                      Field(&otel_metrics::HistogramPointData::counts_,
                            ElementsAreArray(expected.bucket_counts))))),
      Field(&otel_metrics::PointDataAttributes::attributes,
            UnorderedElementsAreArray(expected.attributes.begin(), expected.attributes.end())));
}

auto MatchPointDataVector(const std::vector<ExpectedHist>& expected) {
  using MatchPointDataType = decltype(MatchPointData(expected[0]));

  std::vector<MatchPointDataType> matchers;
  matchers.reserve(expected.size());
  for (const auto& x : expected) {
    matchers.push_back(MatchPointData(x));
  }
  return UnorderedElementsAreArray(matchers);
}

struct PacketAndExpectation {
  std::vector<std::string> input_labels_pbtxts;
  std::vector<ExpectedHist> expected_hists;
};

class ClassificationsMetricsSinkTest
    : public ::testing::TestWithParam<std::vector<PacketAndExpectation>> {};

TEST_P(ClassificationsMetricsSinkTest, CollectsStatsCorrectly) {
  auto packet_and_expectation = GetParam();

  constexpr char kClassificationsMetricsSinkNode[] = R"pbtxt(
    calculator: "ClassificationsMetricsSinkCalculator"
    input_stream: "classifications"
  )pbtxt";

  mediapipe::CalculatorRunner runner(kClassificationsMetricsSinkNode);

  // The test param comes as a packet per time step and an expectation for that timestep.
  for (size_t t = 0; t < packet_and_expectation.size(); ++t) {
    const auto& input_labels_pbtxts = packet_and_expectation[t].input_labels_pbtxts;
    const auto& expected_hists = packet_and_expectation[t].expected_hists;

    // Convert the pbtxts into Label objects.
    std::vector<Label> labels(input_labels_pbtxts.size());
    for (size_t i = 0; i < input_labels_pbtxts.size(); i++) {
      CHECK(google::protobuf::TextFormat::ParseFromString(input_labels_pbtxts[i], &labels[i]));
    }

    mediapipe::Packet p = mediapipe::MakePacket<std::vector<Label>>(std::move(labels));
    p = p.At(mediapipe::Timestamp(static_cast<int64_t>(t)));
    runner.MutableInputs()->Index(0).packets.push_back(p);

    ASSERT_OK(runner.Run());

    auto& metrics_system = metrics::MetricsSystem::GetInstance();

    auto check_results =
        [&expected_hists](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
          const auto& scope_metrics = resource_metrics.scope_metric_data_;
          ASSERT_EQ(1, scope_metrics.size());

          const auto& metric_data = scope_metrics[0].metric_data_;
          ASSERT_EQ(1, metric_data.size());

          for (const auto& metric_datum : metric_data) {
            const auto& point_data = metric_datum.point_data_attr_;
            EXPECT_THAT(point_data, MatchPointDataVector(expected_hists));
          }
        };
    auto results_cb =
        [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
          check_results(resource_metrics);
          return true;
        };
    metrics_system.Reader()->Collect(results_cb);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ClassificationsMetricsSinkTestSuite, ClassificationsMetricsSinkTest,
    ::testing::Values(std::vector<PacketAndExpectation>{
        PacketAndExpectation{
            std::vector<std::string>{
                R"pbtxt(
                  label: "bottle"
                  score: 0.89999
                )pbtxt",
                R"pbtxt(
                  label: "can"
                  score: 0.59999
                )pbtxt",
                R"pbtxt(
                  label: "person"
                  score: 0.19999
                )pbtxt",
            },
            std::vector<ExpectedHist>{
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                    {{"class", "bottle"}, {"k", "1"}},
                },
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {{"class", "can"}, {"k", "2"}},
                },
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {{"class", "person"}, {"k", "3"}},
                },
            },
        },
        PacketAndExpectation{
            std::vector<std::string>{
                R"pbtxt(
                  label: "bottle"
                  score: 0.79999
                )pbtxt",
                R"pbtxt(
                  label: "can"
                  score: 0.58888
                )pbtxt",
                R"pbtxt(
                  label: "wrench"
                  score: 0.19999
                )pbtxt",
                R"pbtxt(
                  label: "hammer"
                  score: 0.09999
                )pbtxt",
            },
            std::vector<ExpectedHist>{
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0},
                    {{"class", "bottle"}, {"k", "1"}},
                },
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {{"class", "can"}, {"k", "2"}},
                },
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {{"class", "person"}, {"k", "3"}},
                },
                ExpectedHist{
                    {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {{"class", "wrench"}, {"k", "3"}},
                },
            },
        },
    }));

}  // namespace gml::gem::calculators::core
