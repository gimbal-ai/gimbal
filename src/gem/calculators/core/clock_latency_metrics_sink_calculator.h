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

#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/base/status.h"
#include "src/gem/calculators/core/metrics_sink_calculator.h"

namespace gml::gem::calculators::core {

/**
 *  ClockLatencyMetricsSinkCalculator Graph API:
 *
 *  Inputs:
 *    mediapipe::Packet<absl::Duration> packet containing latency data.
 *
 *  No outputs, outputs stats to opentelemetry.
 */

class ClockLatencyMetricsSinkCalculator : public MetricsSinkCalculator<absl::Duration> {
 protected:
  Status BuildMetrics(mediapipe::CalculatorContext* cc) override;
  Status RecordMetrics(mediapipe::CalculatorContext* cc) override;

 private:
  opentelemetry::metrics::Histogram<double>* latency_hist_;
};

}  // namespace gml::gem::calculators::core
