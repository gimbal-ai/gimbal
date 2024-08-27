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

struct PacketAndExpectation {
  std::vector<std::string> detection_pbtxts;
  absl::flat_hash_map<std::string, std::vector<ExpectedMetric>> expected_metrics;
};

std::ostream& operator<<(std::ostream& os, const PacketAndExpectation& packet_and_expectation) {
  os << "PacketAndExpectation{";
  os << "detection_pbtxts: {";
  for (const auto& detection_pbtxt : packet_and_expectation.detection_pbtxts) {
    os << detection_pbtxt << ", ";
  }
  os << "}, ";
  os << "}";
  return os;
}

class DetectionsMetricsSinkTest : public ::testing::TestWithParam<PacketAndExpectation> {};

TEST_P(DetectionsMetricsSinkTest, CollectsStatsCorrectly) {
  auto packet_and_expectation = GetParam();

  constexpr char kDetectionsMetricsSinkNode[] = R"pbtxt(
    calculator: "DetectionsMetricsSinkCalculator"
    input_stream: "detection_list"
    output_stream: "FINISHED:finished"
  )pbtxt";

  testing::CalculatorTester tester(kDetectionsMetricsSinkNode);

  const auto& detection_pbtxts = packet_and_expectation.detection_pbtxts;
  auto& expected_hists = packet_and_expectation.expected_metrics;

  std::vector<Detection> detections(detection_pbtxts.size());
  for (size_t i = 0; i < detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(detection_pbtxts[i], &detections[i]));
  }

  tester.ForInput(0, std::move(detections), mediapipe::Timestamp(0))
      .Run()
      .ExpectOutput<bool>("FINISHED", 0, mediapipe::Timestamp(0), true);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  CheckMetrics(metrics_system, expected_hists);
}

INSTANTIATE_TEST_SUITE_P(
    DetectionsMetricsSinkTestSuite, DetectionsMetricsSinkTest,
    ::testing::Values(PacketAndExpectation{
        std::vector<std::string>{
            R"pbtxt(
                label {
                  label: "hammer"
                  score: 0.89999
                }
                label {
                  label: "mallet"
                  score: 0.69999
                }
                label {
                  label: "wrench"
                  score: 0.19999
                }
                label {
                  label: "person"
                  score: 0.09999
                }
                bounding_box {
                  xc: 0.5
                  yc: 0.2
                  width: 0.1
                  height: 0.2
                }
            )pbtxt",
            R"pbtxt(
                label {
                  label: "hammer"
                  score: 0.09999
                }
                bounding_box {
                  xc: 0.5
                  yc: 0.2
                  width: 0.4
                  height: 0.5
                }
            )pbtxt",
            R"pbtxt(
                label {
                  label: "mallet"
                  score: 0.59999
                }
                bounding_box {
                  xc: 0.5
                  yc: 0.2
                  width: 0.1
                  height: 0.01
                }
            )pbtxt",
        },
        ExpectedMetricsMap{
            {
                "gml_gem_pipe_detections_aspect_ratio",
                std::vector<ExpectedMetric>{
                    ExpectedHist{
                        {0, 0.02, 0.2, 0.5, 1, 2, 5, 50},
                        {0, 0, 0, 1, 1, 0, 0, 0, 0},
                        {{"class", std::string("hammer")}},
                    },
                    ExpectedHist{
                        {0, 0.02, 0.2, 0.5, 1, 2, 5, 50},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {{"class", std::string("mallet")}},
                    },
                },
            },
            {
                "gml_gem_pipe_detections_area",
                std::vector<ExpectedMetric>{
                    ExpectedHist{
                        {0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1},
                        {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("hammer")}},
                    },
                    ExpectedHist{
                        {0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("mallet")}},
                    },
                },
            },
            {
                "gml_gem_pipe_detections_confidence",
                std::vector<ExpectedMetric>{
                    ExpectedHist{
                        {0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,
                         0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {{"class", std::string("hammer")}, {"k", std::string("1")}},
                    },
                    ExpectedHist{
                        {0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,
                         0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("mallet")}, {"k", std::string("1")}},
                    },
                    ExpectedHist{
                        {0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,
                         0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("mallet")}, {"k", std::string("2")}},
                    },
                    ExpectedHist{
                        {0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,
                         0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("wrench")}, {"k", std::string("3")}},
                    },

                },
            },
        },
    }));

}  // namespace gml::gem::calculators::core
