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

#include <gmock/gmock.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/core/test_utils.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/core/otel_utils.h"

namespace gml::gem::calculators::core {

TEST(FlowLimiterMetricsSinkCalculatorTest, Basic) {
  constexpr char kGraph[] = R"pbtxt(
      calculator: "FlowLimiterMetricsSinkCalculator"
      input_stream: "allow"
    )pbtxt";

  testing::CalculatorTester tester(kGraph);

  tester.ForInput(0, true, mediapipe::Timestamp(0))
      .ForInput(0, true, mediapipe::Timestamp(1))
      .ForInput(0, false, mediapipe::Timestamp(2))
      .ForInput(0, false, mediapipe::Timestamp(3))
      .ForInput(0, true, mediapipe::Timestamp(4))
      .Run();

  ExpectedMetricsMap expectations{
      {"gml_gem_pipe_flow_limiter_allows", {ExpectedCounter<int64_t>{3}}},
      {"gml_gem_pipe_flow_limiter_drops", {ExpectedCounter<int64_t>{2}}},
  };
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  CheckMetrics(metrics_system, expectations);
}

}  // namespace gml::gem::calculators::core
