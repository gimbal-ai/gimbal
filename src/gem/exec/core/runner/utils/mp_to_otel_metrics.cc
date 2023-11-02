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

#include <mediapipe/framework/calculator_framework.h>

namespace gml {
namespace gem {
namespace utils {

using Metric = ::opentelemetry::proto::metrics::v1::Metric;

namespace {

void PopulateIntSumMetric(Metric* metric, std::string_view name, std::string_view desc,
                          std::string_view unit, int64_t time_since_epoch_ns, int64_t value) {
  metric->set_name(std::string(name));
  metric->set_description(std::string(desc));
  metric->set_unit(std::string(unit));

  auto* sum = metric->mutable_sum();
  sum->set_is_monotonic(true);
  sum->set_aggregation_temporality(
      opentelemetry::proto::metrics::v1::AGGREGATION_TEMPORALITY_CUMULATIVE);

  auto* data_point = sum->add_data_points();
  data_point->set_time_unix_nano(time_since_epoch_ns);
  data_point->set_as_int(value);
}

}  // namespace

Status CalculatorProfileVecToOTelProto(
    const std::vector<mediapipe::CalculatorProfile>& profiles,
    opentelemetry::proto::metrics::v1::ResourceMetrics* metrics_out) {
  // We need a timestamp for the metrics, so calculate the duration since the epoch now.
  // TODO(oazizi): It would be more correct if we timestamped the profiles data when it was
  //               collected, and use that timestamp.
  auto now = std::chrono::steady_clock::now();
  auto time_since_epoch = now.time_since_epoch();
  auto time_since_epoch_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();

  auto* scope_metrics = metrics_out->add_scope_metrics();
  scope_metrics->mutable_scope()->set_name("mediapipe");
  scope_metrics->mutable_scope()->set_version("v0.0.1");

  constexpr std::string_view kMPStatPrefix = "mediapipe_";

  for (const auto& p : profiles) {
    // Populate open_runtime.
    {
      Metric* open_runtime_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kMPStatPrefix, p.name(), "_open_runtime");
      std::string desc = absl::Substitute(
          "The time the mediapipe $0 stage has spent in the Open() call.", p.name());
      std::string unit = "usec";
      PopulateIntSumMetric(open_runtime_metric, name, desc, unit, time_since_epoch_ns,
                           p.open_runtime());
    }

    // Populate close_runtime.
    {
      Metric* close_runtime_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kMPStatPrefix, p.name(), "_close_runtime");
      std::string desc = absl::Substitute(
          "The time the mediapipe $0 stage has spent in the Close() call.", p.name());
      std::string unit = "usec";
      PopulateIntSumMetric(close_runtime_metric, name, desc, unit, time_since_epoch_ns,
                           p.close_runtime());
    }

    // Populate process_runtime histogram.
    // TODO(oazizi): Implement this.

    // Populate process_input_latency histogram.
    // TODO(oazizi): Implement this.

    // Populate process_output_latency histogram.
    // TODO(oazizi): Implement this.
  }

  return Status::OK();
}

}  // namespace utils
}  // namespace gem
}  // namespace gml
