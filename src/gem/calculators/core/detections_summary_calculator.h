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

namespace gml::gem::calculators::core {

/**
 *  DetectionsSummaryCalculator Graph API:
 *
 *  Inputs:
 *    std::vector<internal::api::corepb::v1::Detection> list of detection protos.
 *
 *  No outputs, outputs stats to opentelemetry.
 */
class DetectionsSummaryCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> detection_hist_;
  std::unique_ptr<opentelemetry::metrics::Histogram<double>> confidence_hist_;
};

}  // namespace gml::gem::calculators::core
