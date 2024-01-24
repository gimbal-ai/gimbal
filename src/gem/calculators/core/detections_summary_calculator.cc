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

#include "src/gem/calculators/core/detections_summary_calculator.h"

#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

constexpr size_t kMaxMetricClasses = 80;

Status DetectionsSummaryCalculator::BuildMetrics(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  // TODO(james): make the bounds configurable in CalculatorOptions.
  detection_hist_ = metrics_system.CreateHistogramWithBounds<uint64_t>(
      "gml_gem_model_detection_classes", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  confidence_hist_ = metrics_system.CreateHistogramWithBounds<double>(
      "gml_gem_model_confidence_classes", {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});
  return Status::OK();
}

Status DetectionsSummaryCalculator::RecordMetrics(const std::vector<Detection>& detections) {
  // If there are many labels in a detection, we don't want to send metrics for all of them, as the
  // metrics will not be useful after some point. Instead, only send metrics for the top N
  // classes.
  auto cmp = [](const std::pair<std::string, double>& lhs,
                const std::pair<std::string, double>& rhs) { return lhs.second > rhs.second; };
  std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>,
                      decltype(cmp)>
      top_classes(cmp);

  std::map<std::string, uint64_t> class_counts;

  for (const auto& detection : detections) {
    for (const auto& label : detection.label()) {
      class_counts[label.label()] += 1;
      top_classes.push({label.label(), label.score()});

      if (top_classes.size() > kMaxMetricClasses) {
        top_classes.pop();
      }
    }
  }

  while (top_classes.size() > 0) {
    auto c = top_classes.top();
    top_classes.pop();
    confidence_hist_->Record(c.second, {{"class", c.first}}, {});

    auto class_count = class_counts.find(c.first);
    if (class_count != class_counts.end()) {
      detection_hist_->Record(class_count->second, {{"class", c.first}}, {});
      class_counts.erase(c.first);
    }
  }

  return Status::OK();
}

REGISTER_CALCULATOR(DetectionsSummaryCalculator);

}  // namespace gml::gem::calculators::core
