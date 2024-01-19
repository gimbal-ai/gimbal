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

#include "src/gem/exec/core/runner/utils/mp_to_otel_metrics.h"

#include <fstream>
#include <iostream>

#include <google/protobuf/text_format.h>
#include <mediapipe/framework/profiler/graph_profiler.h>

#include "src/common/base/file.h"
#include "src/common/testing/testing.h"

namespace gml::gem::utils {

// Note: This test input was collected from a real MediaPipe run, but then
// manually modified to have 10 buckets instead of the original 100.
const char kMediapipePB[] = R"(
name: "CountingSourceCalculator"
open_runtime: 37
close_runtime: 0
process_runtime {
  total: 42
  interval_size_usec: 1000
  num_intervals: 10
  count: 11
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
}
process_input_latency {
  total: 0
  interval_size_usec: 1000
  num_intervals: 10
  count: 11
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
}
process_output_latency {
  total: 42
  interval_size_usec: 1000
  num_intervals: 10
  count: 11
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
  count: 0
}
)";

// Note: This expectation is missing the time_unix_nano fields, and so we use a partial
//       PB check in the test below.
// TODO(oazizi): Once the timestamp can be injected, update this test.

// Note: This expectation was created from studying the protobuf spec.
//       It should be validated that this conforms to the metrics consumer's expectations.
const char kExpectedPB[] = R"(
scope_metrics {
  scope {
    name: "gml_gem_exec_mp"
    version: "v0.0.1"
  }
  metrics {
    name: "gml_gem_exec_mp_open_runtime_seconds_total"
    description: "The time mediapipe has spent in the Open() call."
    unit: "seconds"
    sum {
      data_points {
      start_time_unix_nano: 1000
        as_double: 3.7e-05
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
      is_monotonic: true
    }
  }
  metrics {
    name: "gml_gem_exec_mp_close_runtime_seconds_total"
    description: "The time mediapipe has spent in the Close() call."
    unit: "seconds"
    sum {
      data_points {
        start_time_unix_nano: 1000
        as_double: 0
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
      is_monotonic: true
    }
  }
  metrics {
    name: "gml_gem_exec_mp_process_runtime_seconds"
    description: "The time mediapipe has spent in the Process() call."
    unit: "seconds"
    histogram {
      data_points {
        start_time_unix_nano: 1000
        count: 11
        sum: 4.2e-05
        bucket_counts: 11
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        explicit_bounds: 0.001
        explicit_bounds: 0.002
        explicit_bounds: 0.003
        explicit_bounds: 0.004
        explicit_bounds: 0.005
        explicit_bounds: 0.006
        explicit_bounds: 0.007
        explicit_bounds: 0.008
        explicit_bounds: 0.009
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
    }
  }
  metrics {
    name: "gml_gem_exec_mp_process_input_latency_seconds"
    description: "The Process() input latency of mediapipe."
    unit: "seconds"
    histogram {
      data_points {
        start_time_unix_nano: 1000
        count: 11
        sum: 4.2e-05
        bucket_counts: 11
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        explicit_bounds: 0.001
        explicit_bounds: 0.002
        explicit_bounds: 0.003
        explicit_bounds: 0.004
        explicit_bounds: 0.005
        explicit_bounds: 0.006
        explicit_bounds: 0.007
        explicit_bounds: 0.008
        explicit_bounds: 0.009
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
    }
  }
  metrics {
    name: "gml_gem_exec_mp_process_output_latency_seconds"
    description: "The Process() output latency of mediapipe."
    unit: "seconds"
    histogram {
      data_points {
        start_time_unix_nano: 1000
        count: 11
        sum: 4.2e-05
        bucket_counts: 11
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        bucket_counts: 0
        explicit_bounds: 0.001
        explicit_bounds: 0.002
        explicit_bounds: 0.003
        explicit_bounds: 0.004
        explicit_bounds: 0.005
        explicit_bounds: 0.006
        explicit_bounds: 0.007
        explicit_bounds: 0.008
        explicit_bounds: 0.009
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
      }
    }
}
)";

constexpr int64_t kStartTimeUnixNs = 1000;

TEST(MpToOTelMetrics, Basic) {
  ::mediapipe::CalculatorProfile calculator_profile;
  ::google::protobuf::TextFormat::ParseFromString(kMediapipePB, &calculator_profile);

  opentelemetry::proto::metrics::v1::ResourceMetrics metrics;
  EXPECT_OK(CalculatorProfileVecToOTelProto(
      std::vector<mediapipe::CalculatorProfile>{calculator_profile}, kStartTimeUnixNs, &metrics));

  EXPECT_THAT(metrics, testing::proto::Partially(testing::proto::EqualsProto(kExpectedPB)));

  std::string metrics_text;
  google::protobuf::TextFormat::PrintToString(metrics, &metrics_text);
}

TEST(MpToOTelMetrics, MalformedInput) {
  // This PB has a mismatch between num_intervals and number of count fields.
  const char kMediapipePB[] = R"(
    name: "CountingSourceCalculator"
    open_runtime: 10
    close_runtime: 20
    process_runtime {
      total: 30
      interval_size_usec: 1000
      num_intervals: 10
      count: 11
      count: 0
      count: 0
      count: 0
    }
  )";

  ::mediapipe::CalculatorProfile calculator_profile;
  ::google::protobuf::TextFormat::ParseFromString(kMediapipePB, &calculator_profile);

  opentelemetry::proto::metrics::v1::ResourceMetrics metrics;
  EXPECT_NOT_OK(CalculatorProfileVecToOTelProto(
      std::vector<mediapipe::CalculatorProfile>{calculator_profile}, kStartTimeUnixNs, &metrics));
}

}  // namespace gml::gem::utils
