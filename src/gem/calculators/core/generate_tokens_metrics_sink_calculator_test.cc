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

#include "src/gem/calculators/core/generate_tokens_metrics_sink_calculator.h"

#include <mediapipe/framework/formats/detection.pb.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/otel_utils.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kTokenMetricsSinkNode = R"pbtxt(
calculator: "GenerateTokensMetricsSinkCalculator"
node_options {
  [type.googleapis.com/gml.gem.calculators.core.optionspb.GenerateTokensMetricsSinkCalculatorOptions] {
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
input_stream: "INPUT_TOKENS:input_token_ids"
input_stream: "OUTPUT_TOKENS:output_token_ids"
input_stream: "OUTPUT_EOS:eos"
input_stream_handler {
  input_stream_handler: "SyncSetInputStreamHandler"
  options {
    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
      sync_set {
        tag_index: "INPUT_TOKENS"
      }
      sync_set {
        tag_index: "OUTPUT_TOKENS"
        tag_index: "OUTPUT_EOS"
      }
    }
  }
})pbtxt";

struct OutputBatch {
  // Output Streams
  std::vector<int> output_token_ids;
  bool eos;
};

struct InputBatch {
  // Input Streams
  std::vector<int> input_token_ids;
};

struct StreamBatch {
  // Input Streams
  std::optional<InputBatch> input_batch = std::nullopt;
  std::optional<OutputBatch> output_batch = std::nullopt;
  // Metric Values
  absl::flat_hash_map<std::string, ExpectedMetric> expected_metrics;
};

struct GenerateTokenMetricsSinkTestCase {
  std::vector<StreamBatch> stream_batches;
};

class GenerateTokenMetricsSinkTest
    : public ::testing::TestWithParam<GenerateTokenMetricsSinkTestCase> {};

TEST_P(GenerateTokenMetricsSinkTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  std::string config(kTokenMetricsSinkNode);
  testing::CalculatorTester tester(config);
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto ts = mediapipe::Timestamp::Min();
  int64_t test_index = 0;
  for (const auto& batch : test_case.stream_batches) {
    SCOPED_TRACE("For batch " + std::to_string(test_index));
    test_index++;
    if (batch.input_batch.has_value()) {
      tester.ForInput("INPUT_TOKENS", batch.input_batch->input_token_ids, ts);
    }
    if (batch.output_batch.has_value()) {
      tester.ForInput("OUTPUT_TOKENS", batch.output_batch->output_token_ids, ts)
          .ForInput("OUTPUT_EOS", batch.output_batch->eos, ts);
    }
    tester.Run();
    ts = mediapipe::Timestamp(
        ts.Value() +
        static_cast<int64_t>(std::round(mediapipe::Timestamp::kTimestampUnitsPerSecond * 0.3)));

    CheckMetrics(metrics_system, batch.expected_metrics);
    metrics_system.Reset();
  }
}

INSTANTIATE_TEST_SUITE_P(
    GenerateTokenMetricsSinkTestSuite, GenerateTokenMetricsSinkTest,
    ::testing::Values(
        GenerateTokenMetricsSinkTestCase{
            {
                StreamBatch{
                    .input_batch = InputBatch{{1, 2, 3}},
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_input_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3, 2, 1},
                            .eos = false,
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             ExpectedCounter<int64_t>{
                                 3,
                                 {{"pipeline_id", "test_pipeline"}, {"device_id", "test_device"}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {1, 2, 3, 4, 5, 6, 7, 8},
                            .eos = true,
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             ExpectedCounter<int64_t>{11,
                                                      {{"pipeline_id", "test_pipeline"},
                                                       {"device_id", "test_device"}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                            {"gml_gem_pipe_gentokens_output_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                        },
                },
                StreamBatch{
                    .input_batch = InputBatch{{2, 2, 2, 2, 2, 2}},
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3},
                            .eos = false,
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             ExpectedCounter<int64_t>{12,
                                                      {{"pipeline_id", "test_pipeline"},
                                                       {"device_id", "test_device"}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                            {"gml_gem_pipe_gentokens_output_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3},
                            .eos = true,
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             ExpectedCounter<int64_t>{13,
                                                      {{"pipeline_id", "test_pipeline"},
                                                       {"device_id", "test_device"}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                            {"gml_gem_pipe_gentokens_output_hist",
                             ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }},
                        },
                },
            }}));

}  // namespace gml::gem::calculators::core
