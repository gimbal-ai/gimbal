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
#include "src/gem/calculators/core/metrics_utils.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/otel_utils.h"
namespace gml::gem::calculators::core {
const std::vector<double> kThroughputBucketBounds = {0,   0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
                                                     0.1, 0.25,  0.5,    1,     2.5,  5,     10,
                                                     25,  50,    100,    250,   500,  1000,  2500};

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
input_stream: "INPUT_TIMESTAMP:input_timestamp"
input_stream: "INPUT_TOKENS:input_token_ids"
input_stream: "OUTPUT_TOKENS:output_token_ids"
input_stream: "OUTPUT_TIMESTAMP:output_timestamp"
input_stream: "OUTPUT_EOS:eos"
input_stream_handler {
  input_stream_handler: "SyncSetInputStreamHandler"
  options {
    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
      sync_set {
        tag_index: "INPUT_TIMESTAMP"
        tag_index: "INPUT_TOKENS"
      }
      sync_set {
        tag_index: "OUTPUT_TIMESTAMP"
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
  // The duration since the start of the pipeline.
  absl::Duration output_timestamp;
};

struct InputBatch {
  // Input Streams
  std::vector<int> input_token_ids;
  // The duration since the start of the pipeline.
  absl::Duration input_timestamp;
};

struct StreamBatch {
  // Input Streams
  std::optional<InputBatch> input_batch = std::nullopt;
  std::optional<OutputBatch> output_batch = std::nullopt;
  // Metric Values
  ExpectedMetricsMap expected_metrics;
};

struct GenerateTokenMetricsSinkTestCase {
  std::vector<StreamBatch> stream_batches;
};

class GenerateTokenMetricsSinkTest
    : public ::testing::TestWithParam<GenerateTokenMetricsSinkTestCase> {};

TEST_P(GenerateTokenMetricsSinkTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  absl::Time start_time = absl::Now();

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
      tester.ForInput("INPUT_TIMESTAMP", batch.input_batch->input_timestamp + start_time, ts);
    }
    if (batch.output_batch.has_value()) {
      tester.ForInput("OUTPUT_TOKENS", batch.output_batch->output_token_ids, ts)
          .ForInput("OUTPUT_TIMESTAMP", batch.output_batch->output_timestamp + start_time, ts)
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
                    .input_batch = InputBatch{{1, 2, 3}, absl::Seconds(0)},
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3, 2, 1},
                            .eos = false,
                            .output_timestamp = absl::Milliseconds(10),
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{3,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {1, 2, 3, 4, 5, 6, 7, 8},
                            .eos = true,
                            .output_timestamp = absl::Milliseconds(20),
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{11,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_last_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_tokens_per_second_hist",
                             {ExpectedHist{
                                 .bucket_bounds = kThroughputBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
                StreamBatch{
                    .input_batch = InputBatch{{2, 2, 2, 2, 2, 2}, absl::Milliseconds(30)},
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3},
                            .eos = false,
                            .output_timestamp = absl::Milliseconds(31),
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{12,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_last_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_tokens_per_second_hist",
                             {ExpectedHist{
                                 .bucket_bounds = kThroughputBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},

                        },
                },
                // Receieve an input batch before receiving an eos output batch for previous
                // should not update the time_to_first_hist metric
                StreamBatch{
                    .input_batch = InputBatch{{1}, absl::Milliseconds(100)},
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{12,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_last_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_tokens_per_second_hist",
                             {ExpectedHist{
                                 .bucket_bounds = kThroughputBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
                StreamBatch{
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {3},
                            .eos = true,
                            .output_timestamp = absl::Milliseconds(105),
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{13,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            // This metric does not get updated because the input batch
                            // is received before the eos output batch for previous batch
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts =
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_last_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_tokens_per_second_hist",
                             {ExpectedHist{
                                 .bucket_bounds = kThroughputBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
                // Recieves the first output batch right after input batch and record the metric.
                // Also sends the eos and output tokens in the same batch, so tok/s metric does not
                // get updated.
                StreamBatch{
                    .input_batch = InputBatch{{4}, absl::Milliseconds(150)},
                    .output_batch =
                        OutputBatch{
                            .output_token_ids = {4},
                            .eos = true,
                            .output_timestamp = absl::Milliseconds(151),
                        },
                    .expected_metrics =
                        {
                            {"gml_gem_pipe_gentokens_output",
                             {ExpectedCounter<int64_t>{14,
                                                       {{"pipeline_id", "test_pipeline"},
                                                        {"device_id", "test_device"}}}}},
                            {"gml_gem_pipe_gentokens_input_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics::kDefaultHistogramBounds,
                                 .bucket_counts = {0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            // Metric gets updated, for 50ms latency
                            {"gml_gem_pipe_gentokens_time_to_first_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts =
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_time_to_last_hist",
                             {ExpectedHist{
                                 .bucket_bounds = metrics_utils::kLatencySecondsBucketBounds,
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                            {"gml_gem_pipe_gentokens_output_tokens_per_second_hist",
                             {ExpectedHist{
                                 .bucket_bounds = kThroughputBucketBounds,
                                 // Does not get updated because the first output token arrives at
                                 // the same time as eos.
                                 .bucket_counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},

                                 .attributes = {{"pipeline_id", "test_pipeline"},
                                                {"device_id", "test_device"}},
                             }}},
                        },
                },
            }}));

}  // namespace gml::gem::calculators::core
