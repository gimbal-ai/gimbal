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
      auto* open_runtime_metric = scope_metrics->add_metrics();
      open_runtime_metric->set_name(absl::StrCat(kMPStatPrefix, p.name(), "_open_runtime"));
      open_runtime_metric->set_description(absl::Substitute(
          "The time the mediapipe $0 stage has spent in the Open() call.", p.name()));
      open_runtime_metric->set_unit("usec");

      auto* sum = open_runtime_metric->mutable_sum();
      sum->set_is_monotonic(true);
      sum->set_aggregation_temporality(
          opentelemetry::proto::metrics::v1::AGGREGATION_TEMPORALITY_CUMULATIVE);

      auto* data_point = sum->add_data_points();
      data_point->set_time_unix_nano(time_since_epoch_ns);
      data_point->set_as_int(p.open_runtime());
    }

    // Populate close_runtime.
    {
      auto* close_runtime_metric = scope_metrics->add_metrics();
      close_runtime_metric->set_name(absl::StrCat(kMPStatPrefix, p.name(), "_close_runtime"));
      close_runtime_metric->set_description(absl::Substitute(
          "The time the mediapipe $0 stage has spent in the Close() call.", p.name()));
      close_runtime_metric->set_unit("usec");

      auto* sum = close_runtime_metric->mutable_sum();
      sum->set_is_monotonic(true);
      sum->set_aggregation_temporality(
          opentelemetry::proto::metrics::v1::AGGREGATION_TEMPORALITY_CUMULATIVE);

      auto* data_point = sum->add_data_points();
      data_point->set_time_unix_nano(time_since_epoch_ns);
      data_point->set_as_int(p.close_runtime());
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
