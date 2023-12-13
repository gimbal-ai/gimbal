/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
 */

#include "src/gem/calculators/core/detections_summary_calculator.h"
#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

static constexpr char kDetectionsSummaryNode[] = R"pbtxt(
calculator: "DetectionsSummaryCalculator"
input_stream: "detection_list"
)pbtxt";

struct ExpectedHist {
  std::vector<double> bucket_bounds;
  std::vector<uint64_t> bucket_counts;
};

struct DetectionsSummaryTestCase {
  std::vector<std::string> detection_pbtxts;

  std::vector<ExpectedHist> expected_hists;
};

class DetectionsSummaryTest : public ::testing::TestWithParam<DetectionsSummaryTestCase> {};

TEST_P(DetectionsSummaryTest, CollectsStatsCorrectly) {
  auto test_case = GetParam();

  testing::CalculatorTester tester(kDetectionsSummaryNode);

  std::vector<Detection> detections(test_case.detection_pbtxts.size());

  for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                              &detections[i]));
  }

  tester.ForInput(0, std::move(detections), 0).Run();

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto check_results = [&test_case](
                           opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
    const auto& scope_metrics = resource_metrics.scope_metric_data_;
    ASSERT_EQ(1, scope_metrics.size());
    const auto& metric_data = scope_metrics[0].metric_data_;
    ASSERT_EQ(1, metric_data.size());
    const auto& point_data = metric_data[0].point_data_attr_;
    ASSERT_EQ(test_case.expected_hists.size(), point_data.size());
    for (const auto& [i, p] : Enumerate(point_data)) {
      ASSERT_TRUE(
          std::holds_alternative<opentelemetry::sdk::metrics::HistogramPointData>(p.point_data));
      const auto& histogram_point_data =
          std::get<opentelemetry::sdk::metrics::HistogramPointData>(p.point_data);
      EXPECT_EQ(test_case.expected_hists[i].bucket_bounds, histogram_point_data.boundaries_);
      EXPECT_EQ(test_case.expected_hists[i].bucket_counts, histogram_point_data.counts_);
    }
  };
  auto results_cb =
      [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        check_results(resource_metrics);
        return true;
      };
  metrics_system.Reader()->Collect(results_cb);
}

INSTANTIATE_TEST_SUITE_P(DetectionsSummaryTestSuite, DetectionsSummaryTest,
                         ::testing::Values(DetectionsSummaryTestCase{
                             {
                                 R"pbtxt(
label {
  label: "bottle"
  score: 0.9
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
  label: "bottle"
  score: 0.9
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
  label: "person"
  score: 0.9
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
                             },
                             {
                                 {
                                     {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
                                     {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 },
                                 {
                                     {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
                                     {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 },
                             },
                         }));

}  // namespace gml::gem::calculators::core
