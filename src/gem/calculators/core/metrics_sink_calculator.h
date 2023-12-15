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
 *  No outputs, outputs stats to opentelemetry.
 */
template <typename T>
class MetricsSinkCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<T>();
    return absl::OkStatus();
  }
  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(BuildMetrics(cc));
    return absl::OkStatus();
  }
  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(RecordMetrics(cc->Inputs().Index(0).Get<T>()));
    return absl::OkStatus();
  }
  absl::Status Close(mediapipe::CalculatorContext*) override { return absl::OkStatus(); }

 protected:
  virtual Status BuildMetrics(mediapipe::CalculatorContext*) { return Status::OK(); }
  virtual Status RecordMetrics(const T& data) = 0;
};

}  // namespace gml::gem::calculators::core
