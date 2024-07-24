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
#include "src/gem/calculators/core/metrics_sink_calculator.h"

namespace gml::gem::calculators::core {

/**
 *  DetectionsSummaryCalculator Graph API:
 *
 *  Inputs:
 *    std::vector<internal::api::corepb::v1::Detection> list of detection protos.
 *
 *  No real outputs, outputs stats to opentelemetry. Optional bool output to signal when processing
 *    is finished.
 */

class DetectionsSummaryCalculator
    : public MetricsSinkCalculator<std::vector<::gml::internal::api::core::v1::Detection>> {
 protected:
  Status BuildMetrics(mediapipe::CalculatorContext* cc) override;
  Status RecordMetrics(mediapipe::CalculatorContext* cc) override;

 private:
  std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> detection_hist_;
  std::unique_ptr<opentelemetry::metrics::Histogram<double>> confidence_hist_;
  std::unique_ptr<opentelemetry::metrics::Histogram<double>> box_area_hist_;
  std::unique_ptr<opentelemetry::metrics::Histogram<double>> box_aspect_ratio_hist_;
};

}  // namespace gml::gem::calculators::core
