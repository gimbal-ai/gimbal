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

#include "src/gem/calculators/core/detections_summary_calculator.h"

#include "gmock/gmock.h"
#include <absl/container/flat_hash_map.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

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

struct DetectionsSummaryTestCase {
  std::vector<std::string> detection_pbtxts;

  absl::flat_hash_map<std::string, std::vector<ExpectedHist>> expected_hists;
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
      EXPECT_THAT(point_data, ::testing::UnorderedElementsAre(MatchHistogram(expected_hists[0]),
                                                              MatchHistogram(expected_hists[1])));
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
  width: 0.4
  height: 0.5
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
  height: 0.01
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
            {
                "gml_gem_model_box_area",
                {
                    {
                        {0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0},
                        {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "bottle"}},
                    },
                    {
                        {0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "person"}},
                    },
                },
            },
            {
                "gml_gem_model_box_aspect_ratio",
                {
                    {

                        {0, 0.02, 0.2, 0.5, 1, 2.0, 5, 50},
                        {0, 0, 0, 1, 1, 0, 0, 0, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "bottle"}},
                    },
                    {
                        {0, 0.02, 0.2, 0.5, 1, 2.0, 5, 50},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {{"pipeline_id", "test_pipeline"},
                         {"device_id", "test_device"},
                         {"class", "person"}},
                    },
                },
            },
        }},
    }));

}  // namespace gml::gem::calculators::core
