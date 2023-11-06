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

#include <vector>

#include "src/gem/exec/core/runner/utils/mp_to_otel_metrics.h"

#include <mediapipe/framework/calculator_framework.h>

namespace gml {
namespace gem {
namespace utils {

using Metric = ::opentelemetry::proto::metrics::v1::Metric;

namespace {

constexpr int64_t kUSecondsPerSecond = 1000 * 1000;

Status PopulateSumMetric(Metric* metric, std::string&& name, std::string&& desc, std::string&& unit,
                         absl::flat_hash_map<std::string, std::string>&& attributes,
                         int64_t time_since_epoch_ns, int64_t value) {
  metric->set_name(std::move(name));
  metric->set_description(std::move(desc));
  metric->set_unit(std::move(unit));

  auto* sum = metric->mutable_sum();
  sum->set_is_monotonic(true);
  sum->set_aggregation_temporality(
      opentelemetry::proto::metrics::v1::AGGREGATION_TEMPORALITY_CUMULATIVE);

  auto* data_point = sum->add_data_points();
  data_point->set_time_unix_nano(time_since_epoch_ns);
  data_point->set_as_double(static_cast<double>(value) / kUSecondsPerSecond);

  for (auto [k, v] : attributes) {
    auto* attribute = data_point->add_attributes();
    attribute->set_key(k);
    attribute->mutable_value()->set_string_value(std::move(v));
  }

  return Status::OK();
}

Status PopulateHistogramMetric(Metric* metric, std::string&& name, std::string&& desc,
                               std::string&& unit,
                               absl::flat_hash_map<std::string, std::string>&& attributes,
                               int64_t time_since_epoch_ns,
                               const ::mediapipe::TimeHistogram& value) {
  metric->set_name(std::move(name));
  metric->set_description(std::move(desc));
  metric->set_unit(std::move(unit));

  auto* histogram = metric->mutable_histogram();
  histogram->set_aggregation_temporality(
      opentelemetry::proto::metrics::v1::AGGREGATION_TEMPORALITY_CUMULATIVE);

  auto* data_point = histogram->add_data_points();
  data_point->set_time_unix_nano(time_since_epoch_ns);

  data_point->set_sum(static_cast<double>(value.total()) / kUSecondsPerSecond);

  if (value.num_intervals() != value.count_size()) {
    return error::Internal(
        "Inconsistency in sizes: num_interval=$0 count_size=$1. Metric histogram is incomplete.",
        value.num_intervals(), value.count_size());
  }

  int64_t interval = value.interval_size_usec();
  int64_t current_boundary = 0;
  for (int i = 0; i < value.num_intervals() - 1; ++i) {
    current_boundary += interval;
    data_point->add_explicit_bounds(static_cast<double>(current_boundary) / kUSecondsPerSecond);
  }

  int64_t total_count = 0;
  for (const auto& count : value.count()) {
    data_point->add_bucket_counts(count);
    total_count += count;
  }
  data_point->set_count(total_count);

  for (auto [k, v] : attributes) {
    auto* attribute = data_point->add_attributes();
    attribute->set_key(k);
    attribute->mutable_value()->set_string_value(std::move(v));
  }

  return Status::OK();
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
  // Scope information will be ignored by our metrics store, but setting for completeness.
  scope_metrics->mutable_scope()->set_name("gml_gem_exec_mp");
  scope_metrics->mutable_scope()->set_version("v0.0.1");

  constexpr std::string_view kStatPrefix = "gml_gem_exec_mp_";

  Status status;

  for (const auto& p : profiles) {
    const std::pair<std::string, std::string> kStageNameAttr{"stage", p.name()};

    // Populate open_runtime.
    {
      Metric* open_runtime_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kStatPrefix, "open_runtime_seconds_total");
      std::string desc = "The time mediapipe has spent in the Open() call.";
      std::string unit = "seconds";
      absl::flat_hash_map<std::string, std::string> attributes{kStageNameAttr};
      GML_RETURN_IF_ERROR(PopulateSumMetric(open_runtime_metric, std::move(name), std::move(desc),
                                            std::move(unit), std::move(attributes),
                                            time_since_epoch_ns, p.open_runtime()));
    }

    // Populate close_runtime.
    {
      Metric* close_runtime_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kStatPrefix, "close_runtime_seconds_total");
      std::string desc = "The time mediapipe has spent in the Close() call.";
      std::string unit = "seconds";
      absl::flat_hash_map<std::string, std::string> attributes{kStageNameAttr};
      GML_RETURN_IF_ERROR(PopulateSumMetric(close_runtime_metric, std::move(name), std::move(desc),
                                            std::move(unit), std::move(attributes),
                                            time_since_epoch_ns, p.close_runtime()));
    }

    // Populate process_runtime histogram.
    {
      Metric* process_runtime_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kStatPrefix, "process_runtime_seconds");
      std::string desc = "The time mediapipe has spent in the Process() call.";
      std::string unit = "seconds";
      absl::flat_hash_map<std::string, std::string> attributes{kStageNameAttr};
      GML_RETURN_IF_ERROR(PopulateHistogramMetric(
          process_runtime_metric, std::move(name), std::move(desc), std::move(unit),
          std::move(attributes), time_since_epoch_ns, p.process_runtime()));
    }

    // Populate process_input_latency histogram.
    {
      Metric* process_input_latency_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kStatPrefix, "process_input_latency_seconds");
      std::string desc = "The Process() input latency of mediapipe.";
      std::string unit = "seconds";
      absl::flat_hash_map<std::string, std::string> attributes{kStageNameAttr};
      GML_RETURN_IF_ERROR(PopulateHistogramMetric(
          process_input_latency_metric, std::move(name), std::move(desc), std::move(unit),
          std::move(attributes), time_since_epoch_ns, p.process_runtime()));
    }

    // Populate process_output_latency histogram.
    {
      Metric* process_output_latency_metric = scope_metrics->add_metrics();
      std::string name = absl::StrCat(kStatPrefix, "process_output_latency_seconds");
      std::string desc = "The Process() output latency of mediapipe.";
      std::string unit = "seconds";
      absl::flat_hash_map<std::string, std::string> attributes{kStageNameAttr};
      GML_RETURN_IF_ERROR(PopulateHistogramMetric(
          process_output_latency_metric, std::move(name), std::move(desc), std::move(unit),
          std::move(attributes), time_since_epoch_ns, p.process_runtime()));
    }
  }

  return Status::OK();
}

}  // namespace utils
}  // namespace gem
}  // namespace gml
