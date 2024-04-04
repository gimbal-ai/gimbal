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
#include "src/gem/calculators/core/optionspb/detections_summary_calculator_options.pb.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

constexpr size_t kMaxMetricClasses = 80;

const std::vector<double> kDetectionClassesBounds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const std::vector<double> kConfidenceClassesBounds = {0,   0.1, 0.2, 0.3, 0.4,
                                                      0.5, 0.6, 0.7, 0.8, 0.9};

Status DetectionsSummaryCalculator::BuildMetrics(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  detection_hist_ = metrics_system.CreateHistogramWithBounds<uint64_t>(
      "gml_gem_model_detection_classes", kDetectionClassesBounds);
  confidence_hist_ = metrics_system.CreateHistogramWithBounds<double>(
      "gml_gem_model_confidence_classes", kConfidenceClassesBounds);
  return Status::OK();
}

Status DetectionsSummaryCalculator::RecordMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::DetectionsSummaryCalculatorOptions>();
  auto& detections = cc->Inputs().Index(0).Get<std::vector<Detection>>();
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
    std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                       options.metric_attributes().end());
    attrs["class"] = c.first;
    top_classes.pop();
    confidence_hist_->Record(c.second, attrs, {});

    auto class_count = class_counts.find(c.first);
    if (class_count != class_counts.end()) {
      detection_hist_->Record(class_count->second, attrs, {});
      class_counts.erase(c.first);
    }
  }

  return Status::OK();
}

REGISTER_CALCULATOR(DetectionsSummaryCalculator);

}  // namespace gml::gem::calculators::core
