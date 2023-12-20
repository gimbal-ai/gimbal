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

#include "src/gem/calculators/core/packet_latency_metrics_sink_calculator.h"

#include <vector>

#include <mediapipe/calculators/util/latency.pb.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/packet_latency_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

namespace {
std::vector<double> MillisToSeconds(std::vector<double> in) {
  for (double& i : in) {
    i /= 1000.0;
  }
  return in;
}
}  // namespace

Status PacketLatencyMetricsSinkCalculator::BuildMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::PacketLatencyMetricsSinkCalculatorOptions>();
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  // TODO(james): make the bounds configurable in CalculatorOptions.
  latency_hist_ = metrics_system.CreateHistogramWithBounds<double>(
      absl::Substitute("gml_gem_$0_latency_seconds", options.name()),
      MillisToSeconds({0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 150}));
  return Status::OK();
}

Status PacketLatencyMetricsSinkCalculator::RecordMetrics(
    const mediapipe::PacketLatency& packet_latency) {
  latency_hist_->Record(static_cast<double>(packet_latency.current_latency_usec()) / 1000.0 /
                        1000.0);
  return Status::OK();
}

REGISTER_CALCULATOR(PacketLatencyMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core