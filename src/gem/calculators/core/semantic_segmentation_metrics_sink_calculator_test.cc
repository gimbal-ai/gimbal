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

#include <iostream>
#include <vector>

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

using ::gml::internal::api::core::v1::Segmentation;

// Define bucket bounds at the top of the file
const std::vector<double> kAreaPercentageBucketBounds = {0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,
                                                         0.35, 0.4,  0.45, 0.5,  0.55, 0.6,  0.65,
                                                         0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1};

struct PacketAndExpectation {
  std::string segmentation_pbtxt;
  ExpectedMetricsMap expected_metrics;
};

std::ostream& operator<<(std::ostream& os, const PacketAndExpectation& packet_and_expectation) {
  os << "PacketAndExpectation{";
  os << "segmentation_pbtxt: {";
  os << packet_and_expectation.segmentation_pbtxt << ", ";
  os << "}, ";
  os << "}";
  return os;
}

class SegmentationMetricsSinkTest : public ::testing::TestWithParam<PacketAndExpectation> {};

TEST_P(SegmentationMetricsSinkTest, CollectsStatsCorrectly) {
  auto packet_and_expectation = GetParam();

  constexpr char kSemanticSegmentationMetricsSinkNode[] = R"pbtxt(
    calculator: "SemanticSegmentationMetricsSinkCalculator"
    input_stream: "segmentation_list"
    output_stream: "FINISHED:finished"
  )pbtxt";

  testing::CalculatorTester tester(kSemanticSegmentationMetricsSinkNode);

  const auto& segmentation_pbtxt = packet_and_expectation.segmentation_pbtxt;
  auto& expected_metrics = packet_and_expectation.expected_metrics;

  Segmentation segmentation;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(segmentation_pbtxt, &segmentation));

  tester.ForInput(0, std::move(segmentation), mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<bool>("FINISHED", 0, mediapipe::Timestamp(0), true);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  CheckMetrics(metrics_system, expected_metrics);
}

INSTANTIATE_TEST_SUITE_P(SegmentationMetricsSinkTestSuite, SegmentationMetricsSinkTest,
                         ::testing::Values(PacketAndExpectation{
                             R"pbtxt(
                              width: 4
                              height: 2
                              masks {
                                label: "person"
                                run_length_encoding: [0, 2, 6, 0]
                              }
                              masks {
                                label: "cat"
                                run_length_encoding: [2, 1, 3, 2]
                              }
                              masks {
                                label: "dog"
                                run_length_encoding: [3, 3, 2, 0]
                              }
                            )pbtxt",
                             {{"gml_gem_pipe_segmentation_area_percentage",
                               {
                                   ExpectedHist{
                                       .bucket_bounds = kAreaPercentageBucketBounds,
                                       .bucket_counts = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       .attributes = {{"class", std::string("person")}},
                                   },
                                   ExpectedHist{
                                       .bucket_bounds = kAreaPercentageBucketBounds,
                                       .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       .attributes = {{"class", std::string("cat")}},
                                   },
                                   ExpectedHist{
                                       .bucket_bounds = kAreaPercentageBucketBounds,
                                       .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       .attributes = {{"class", std::string("dog")}},
                                   },
                               }}},
                         }));

}  // namespace gml::gem::calculators::core
