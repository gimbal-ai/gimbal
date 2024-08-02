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
// Box area is a percentage of the image area. Because area is squared, it is better to create
// buckets as increments of squares of the image area. The second bin is equivalent to a 64x64 box
// in a 640x640 image.
const std::vector<double> kBoxAreaBounds = {0,    0.01, 0.04, 0.09, 0.16, 0.25,
                                            0.36, 0.49, 0.64, 0.81, 1.0};

// Box aspect ratio is the width / height. Bounds are roughly symmetric around 1.
const std::vector<double> kBoxAspectRatioBounds = {0, 0.02, 0.2, 0.5, 1, 2.0, 5, 50};

Status DetectionsSummaryCalculator::BuildMetrics(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  detection_hist_ = metrics_system.GetOrCreateHistogramWithBounds<uint64_t>(
      "gml_gem_model_detection_classes", "Frequency of detection classes predicted by the model.",
      kDetectionClassesBounds);
  confidence_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_model_confidence_classes", "Confidence scores of model predictions.",
      kConfidenceClassesBounds);
  box_area_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_model_box_area",
      "Area of the detection bounding box as a percentage of the image area.", kBoxAreaBounds);
  box_aspect_ratio_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_model_box_aspect_ratio",
      "Aspect ratio (width / height) of the detection bounding box.", kBoxAspectRatioBounds);
  return Status::OK();
}

Status DetectionsSummaryCalculator::RecordMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::DetectionsSummaryCalculatorOptions>();
  auto& detections = cc->Inputs().Index(0).Get<std::vector<Detection>>();
  // If there are many labels in a detection, we don't want to send metrics for all of them, as the
  // metrics will not be useful after some point. Instead, only send metrics for the top N
  // classes.
  struct DetectionInfo {
    std::string label;
    double score;
    double area;
    double aspect_ratio;
  };

  auto cmp = [](const DetectionInfo& lhs, const DetectionInfo& rhs) {
    return lhs.score > rhs.score;
  };
  std::priority_queue<DetectionInfo, std::vector<DetectionInfo>, decltype(cmp)> top_classes(cmp);

  std::map<std::string, uint64_t> class_counts;

  for (const auto& detection : detections) {
    for (const auto& label : detection.label()) {
      class_counts[label.label()] += 1;
      // Box area is a percentage of the image area.
      double box_area = detection.bounding_box().width() * detection.bounding_box().height();
      double aspect_ratio = detection.bounding_box().width() / detection.bounding_box().height();
      top_classes.push({label.label(), label.score(), box_area, aspect_ratio});

      if (top_classes.size() > kMaxMetricClasses) {
        top_classes.pop();
      }
    }
  }

  while (top_classes.size() > 0) {
    auto c = top_classes.top();
    std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                       options.metric_attributes().end());
    attrs["class"] = c.label;
    top_classes.pop();
    confidence_hist_->Record(c.score, attrs, {});
    box_area_hist_->Record(c.area, attrs, {});
    box_aspect_ratio_hist_->Record(c.aspect_ratio, attrs, {});

    auto class_count = class_counts.find(c.label);
    if (class_count != class_counts.end()) {
      detection_hist_->Record(class_count->second, attrs, {});
      class_counts.erase(c.label);
    }
  }

  return Status::OK();
}

REGISTER_CALCULATOR(DetectionsSummaryCalculator);

}  // namespace gml::gem::calculators::core
