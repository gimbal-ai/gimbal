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

#include "src/gem/calculators/core/classification_metrics_sink_calculator.h"

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

using ::gml::internal::api::core::v1::Classification;

struct PacketAndExpectation {
  std::string classification_pbtxt;
  ExpectedMetricsMap expected_metrics;
};

class ClassificationMetricsSinkTest
    : public ::testing::TestWithParam<std::vector<PacketAndExpectation>> {};

TEST_P(ClassificationMetricsSinkTest, CollectsStatsCorrectly) {
  auto packet_and_expectation = GetParam();

  constexpr char kClassificationMetricsSinkNode[] = R"pbtxt(
    calculator: "ClassificationMetricsSinkCalculator"
    input_stream: "classifications"
  )pbtxt";

  mediapipe::CalculatorRunner runner(kClassificationMetricsSinkNode);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  // The test param comes as a packet per time step and an expectation for that timestep.
  for (size_t t = 0; t < packet_and_expectation.size(); ++t) {
    const auto& classification_pbtxt = packet_and_expectation[t].classification_pbtxt;
    const auto& expected_hists = packet_and_expectation[t].expected_metrics;

    // Convert the pbtxts into Label objects.
    Classification classification;
    CHECK(google::protobuf::TextFormat::ParseFromString(classification_pbtxt, &classification));

    mediapipe::Packet p = mediapipe::MakePacket<Classification>(classification);
    p = p.At(mediapipe::Timestamp(static_cast<int64_t>(t)));
    runner.MutableInputs()->Index(0).packets.push_back(p);

    metrics_system.Reset();
    ASSERT_OK(runner.Run());
    CheckMetrics(metrics_system, expected_hists);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ClassificationMetricsSinkTestSuite, ClassificationMetricsSinkTest,
    ::testing::Values(std::vector<PacketAndExpectation>{
        PacketAndExpectation{
            std::string{
                R"pbtxt(
                label: {
                  label: "bottle"
                  score: 0.89999
                },
                label: {
                  label: "can"
                  score: 0.59999
                },
                label: {
                  label: "person"
                  score: 0.19999
                }
                )pbtxt",
            },
            ExpectedMetricsMap{{
                "gml_gem_pipe_classifications_scores",
                std::vector<ExpectedMetric>{
                    ExpectedHist{
                        {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {{"class", std::string("bottle")}, {"k", std::string("1")}},
                    },
                    ExpectedHist{
                        {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("can")}, {"k", std::string("2")}},
                    },
                    ExpectedHist{
                        {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"class", std::string("person")}, {"k", std::string("3")}},
                    },
                },
            }}},
        PacketAndExpectation{std::string{
                                 R"pbtxt(
                label: {
                  label: "bottle"
                  score: 0.79999
                },
                label: {
                  label: "can"
                  score: 0.58888
                },
                label: {
                  label: "wrench"
                  score: 0.19999
                },
                label: {
                  label: "hammer"
                  score: 0.09999
                }
                )pbtxt",
                             },
                             ExpectedMetricsMap{{
                                 "gml_gem_pipe_classifications_scores",
                                 std::vector<ExpectedMetric>{
                                     ExpectedHist{
                                         {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                          0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0},
                                         {{"class", std::string("bottle")},
                                          {"k", std::string("1")}},
                                     },
                                     ExpectedHist{
                                         {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                          0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {{"class", std::string("can")}, {"k", std::string("2")}},
                                     },
                                     ExpectedHist{
                                         {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                          0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                                         {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {{"class", std::string("person")},
                                          {"k", std::string("3")}},
                                     },
                                     ExpectedHist{
                                         {0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                          0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0},
                                         {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {{"class", std::string("wrench")},
                                          {"k", std::string("3")}},
                                     },
                                 },
                             }}},
    }));

}  // namespace gml::gem::calculators::core
