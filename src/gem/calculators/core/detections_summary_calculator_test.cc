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

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

using ::opentelemetry::sdk::common::OwnedAttributeValue;

static constexpr char kDetectionsSummaryNode[] = R"pbtxt(
calculator: "DetectionsSummaryCalculator"
node_options {
  [type.googleapis.com/gml.gem.calculators.core.optionspb.DetectionsSummaryCalculatorOptions] {
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
input_stream: "detection_list"
output_stream: "FINISHED:finished"
)pbtxt";

struct ExpectedHist {
  std::vector<double> bucket_bounds;
  std::vector<uint64_t> bucket_counts;
  absl::flat_hash_map<std::string, OwnedAttributeValue> attributes;
};

struct DetectionsSummaryTestCase {
  std::vector<std::string> detection_pbtxts;

  absl::flat_hash_map<std::string, std::vector<ExpectedHist>> expected_hists;
};

class DetectionsSummaryTest : public ::testing::TestWithParam<DetectionsSummaryTestCase> {};

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

TEST_P(DetectionsSummaryTest, CollectsStatsCorrectly) {
  auto test_case = GetParam();

  testing::CalculatorTester tester(kDetectionsSummaryNode);

  std::vector<Detection> detections(test_case.detection_pbtxts.size());

  for (size_t i = 0; i < test_case.detection_pbtxts.size(); i++) {
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(test_case.detection_pbtxts[i],
                                                              &detections[i]));
  }

  tester.ForInput(0, std::move(detections), mediapipe::Timestamp::Min())
      .Run()
      .ExpectOutput<bool>("FINISHED", 0, mediapipe::Timestamp::Min(), true);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto check_results = [&test_case](
                           opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
    const auto& scope_metrics = resource_metrics.scope_metric_data_;
    ASSERT_EQ(1, scope_metrics.size());

    const auto& metric_data = scope_metrics[0].metric_data_;
    ASSERT_EQ(test_case.expected_hists.size(), metric_data.size());

    for (const auto& metric_datum : metric_data) {
      const auto& name = metric_datum.instrument_descriptor.name_;
      ASSERT_TRUE(test_case.expected_hists.contains(name));
      const auto& expected_hists = test_case.expected_hists[name];
      const auto& point_data = metric_datum.point_data_attr_;
      ASSERT_EQ(expected_hists.size(), point_data.size());
      EXPECT_THAT(point_data, ::testing::UnorderedElementsAre(MatchPointData(expected_hists[0]),
                                                              MatchPointData(expected_hists[1])));
    }
  };
  auto results_cb =
      [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        check_results(resource_metrics);
        return true;
      };
  metrics_system.Reader()->Collect(results_cb);
}

INSTANTIATE_TEST_SUITE_P(
    DetectionsSummaryTestSuite, DetectionsSummaryTest,
    ::testing::Values(DetectionsSummaryTestCase{
        {
            R"pbtxt(
label {
  label: "bottle"
  score: 0.89999
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
  label: "person"
  score: 0.59999
}
bounding_box {
  xc: 0.5
  yc: 0.2
  width: 0.1
  height: 0.2
}
)pbtxt",
        },
        {{
            {
                "gml_gem_model_detection_classes",
                {
                    {
                        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "bottle"}},
                    },
                    {
                        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "person"}},
                    },
                },
            },
            {
                "gml_gem_model_confidence_classes",
                {
                    {
                        {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "bottle"}},
                    },
                    {
                        {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
                        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "person"}},
                    },
                },
            },
        }},
    }));

}  // namespace gml::gem::calculators::core
