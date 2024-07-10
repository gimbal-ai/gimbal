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

#include "src/common/base/status.h"

namespace gml::gem::calculators::core {

/**
 *  MetricsSinkCalculator Graph API:
 *
 *  Inputs:
 *  - T
 *
 *  No real outputs, outputs stats to opentelemetry. Optional bool output to signal processing has
 *    finished.
 */
template <typename T>
class MetricsSinkCalculator : public mediapipe::CalculatorBase {
 private:
  static constexpr std::string_view kFinishedTag = "FINISHED";

 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<T>();
    if (cc->Outputs().HasTag(kFinishedTag)) {
      cc->Outputs().Tag(kFinishedTag).Set<bool>();
    }
    cc->SetTimestampOffset(0);
    return absl::OkStatus();
  }
  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(BuildMetrics(cc));
    return absl::OkStatus();
  }
  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(RecordMetrics(cc));
    if (cc->Outputs().HasTag(kFinishedTag)) {
      cc->Outputs()
          .Tag(kFinishedTag)
          .AddPacket(mediapipe::MakePacket<bool>(true).At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }
  absl::Status Close(mediapipe::CalculatorContext*) override { return absl::OkStatus(); }

 protected:
  virtual Status BuildMetrics(mediapipe::CalculatorContext*) { return Status::OK(); }
  virtual Status RecordMetrics(mediapipe::CalculatorContext*) = 0;
};

}  // namespace gml::gem::calculators::core
