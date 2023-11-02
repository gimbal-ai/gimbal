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

#include <fstream>
#include <iostream>

#include <google/protobuf/text_format.h>

#include <mediapipe/framework/profiler/graph_profiler.h>

#include "src/common/base/file.h"
#include "src/common/testing/testing.h"
#include "src/gem/exec/core/runner/utils/mp_to_otel_metrics.h"

namespace gml {
namespace gem {
namespace utils {

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
const char kExpectedPB[] = R"(
scope_metrics {
  scope {
    name: "mediapipe"
    version: "v0.0.1"
  }
  metrics {
    name: "mediapipe_CountingSourceCalculator_open_runtime"
    description: "The time the mediapipe CountingSourceCalculator stage has spent in the Open() call."
    unit: "usec"
    sum {
      data_points {
        as_int: 37
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
      is_monotonic: true
    }
  }
  metrics {
    name: "mediapipe_CountingSourceCalculator_close_runtime"
    description: "The time the mediapipe CountingSourceCalculator stage has spent in the Close() call."
    unit: "usec"
    sum {
      data_points {
        as_int: 0
      }
      aggregation_temporality: AGGREGATION_TEMPORALITY_CUMULATIVE
      is_monotonic: true
    }
  }
}
)";

TEST(MpToOTelMetrics, Basic) {
  ::mediapipe::CalculatorProfile calculator_profile;
  ::google::protobuf::TextFormat::ParseFromString(kMediapipePB, &calculator_profile);

  opentelemetry::proto::metrics::v1::ResourceMetrics metrics;
  EXPECT_OK(CalculatorProfileVecToOTelProto(
      std::vector<mediapipe::CalculatorProfile>{calculator_profile}, &metrics));

  EXPECT_THAT(metrics, testing::proto::Partially(testing::proto::EqualsProto(kExpectedPB)));

  std::string metrics_text;
  google::protobuf::TextFormat::PrintToString(metrics, &metrics_text);
}

}  // namespace utils
}  // namespace gem
}  // namespace gml
