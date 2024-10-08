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

#pragma once

#include <mediapipe/framework/calculator_framework.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/status.h"

namespace gml::gem::calculators::core {
/**
 *  GenerateTokensMetricsSinkCalculator Graph API:
 *
 *  Note that this is expected to be used with a SyncSetInputStreamHandler on each of the input
 *  streams. SyncSet1 should be {INPUT_TIMESTAMP, INPUT_TOKENS} and SyncSet2 should be
 *  {OUTPUT_TIMESTAMP, OUTPUT_TOKENS, OUTPUT_EOS}.
 *
 *  Inputs:
 *    INPUT_TIMESTAMP absl::Time - The timestamp of the input tokens.
 *    INPUT_TOKENS std::vector<int> - Token IDs encoded from the input stream. Unlike the
 *      output equivalent, this is a single batch that implicitly contains all the input tokens
 *      and therefore does not need an eos.
 *    OUTPUT_TIMESTAMP absl::Time - The timestamp of the output tokens.
 *    OUTPUT_TOKENS std::vector<int> - Token IDs to be written to the output stream.
 *    OUTPUT_EOS bool - A flag indicating whether it is the end of the stream.
 *
 *  Outputs:
 *    This is a sink node so there are no data mediapipe outputs, but will write
 *    stats to opentelemetry.
 */
class GenerateTokensMetricsSinkCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);

  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  void HandleInput(mediapipe::CalculatorContext* cc);
  void HandleOutput(mediapipe::CalculatorContext* cc);

  opentelemetry::metrics::Histogram<uint64_t>* input_tokens_histogram_;
  // We have separate metrics for output histogram and counter because
  // we want to capture "live" metrics for output tokens and also
  // track the full length of output streams, which requires you to accumulate
  // the output stream_count.
  opentelemetry::metrics::Histogram<uint64_t>* output_tokens_histogram_;
  opentelemetry::metrics::Counter<uint64_t>* output_tokens_counter_;
  opentelemetry::metrics::Histogram<double>* time_to_first_output_token_histogram_;
  opentelemetry::metrics::Histogram<double>* time_to_last_output_token_histogram_;
  opentelemetry::metrics::Histogram<double>* output_tokens_per_second_histogram_;
  uint64_t running_output_tokens_ = 0;
  absl::flat_hash_map<std::string, std::string> metric_attributes_;
  struct InputTimestampData {
    absl::Time first_input_token_ts;
    std::optional<absl::Time> first_output_token_ts = std::nullopt;
  };
  std::deque<InputTimestampData> input_timestamps_;
};

}  // namespace gml::gem::calculators::core
