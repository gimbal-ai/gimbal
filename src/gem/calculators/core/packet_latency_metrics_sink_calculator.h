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

#include <mediapipe/calculators/util/latency.pb.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/base/status.h"
#include "src/gem/calculators/core/metrics_sink_calculator.h"

namespace gml::gem::calculators::core {

/**
 *  DetectionsSummaryCalculator Graph API:
 *
 *  Inputs:
 *    std::vector<internal::api::corepb::v1::Detection> list of detection protos.
 *
 *  No outputs, outputs stats to opentelemetry.
 */

class PacketLatencyMetricsSinkCalculator : public MetricsSinkCalculator<mediapipe::PacketLatency> {
 protected:
  Status BuildMetrics(mediapipe::CalculatorContext* cc) override;
  Status RecordMetrics(const mediapipe::PacketLatency& detections) override;

 private:
  std::unique_ptr<opentelemetry::metrics::Histogram<double>> latency_hist_;
};

}  // namespace gml::gem::calculators::core
