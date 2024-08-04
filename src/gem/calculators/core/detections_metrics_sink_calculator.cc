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

#include "src/gem/calculators/core/detections_metrics_sink_calculator.h"

#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/detections_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Detection;

// TODO(oazizi): Consolidate some of these constants with the ones in the ClassificationMetricsSink.

// Maximum number of top classes to report statistics for.
constexpr size_t kMaxK = 3;

// Confidence histogram step size.
constexpr size_t kConfidenceHistogramPctStep = 5;

const std::vector<double> kConfidenceClassesBounds = []() {
  std::vector<double> temp;
  for (int i = 0; i <= 100; i += kConfidenceHistogramPctStep) {
    temp.push_back(i / 100.0);
  }
  return temp;
}();

// Box area is a percentage of the image area. Because area is squared, it is better to create
// buckets as increments of squares of the image area. The second bin is equivalent to a 64x64 box
// in a 640x640 image.
const std::vector<double> kBoxAreaBounds = {0,    0.01, 0.04, 0.09, 0.16, 0.25,
                                            0.36, 0.49, 0.64, 0.81, 1.0};

// Box aspect ratio is the width / height. Bounds are roughly symmetric around 1.
const std::vector<double> kBoxAspectRatioBounds = {0, 0.02, 0.2, 0.5, 1, 2.0, 5, 50};

Status DetectionsMetricsSinkCalculator::BuildMetrics(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  confidence_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_pipe_detections_confidence", "Confidence scores of model predictions.",
      kConfidenceClassesBounds);

  box_area_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_pipe_detections_area",
      "Area of the detection bounding box as a percentage of the image area.", kBoxAreaBounds);

  box_aspect_ratio_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_pipe_detections_aspect_ratio",
      "Aspect ratio (width / height) of the detection bounding box.", kBoxAspectRatioBounds);
  return Status::OK();
}

Status DetectionsMetricsSinkCalculator::RecordMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::DetectionsMetricsSinkCalculatorOptions>();
  auto& detections = cc->Inputs().Index(0).Get<std::vector<Detection>>();

  auto cmp = [](const std::pair<std::string, double>& lhs,
                const std::pair<std::string, double>& rhs) { return lhs.second > rhs.second; };

  std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>,
                      decltype(cmp)>
      top_k_classes(cmp);

  for (const auto& detection : detections) {
    DCHECK(top_k_classes.empty());

    // Box area is a percentage of the image area.
    // Note that the box is normalized to [0, 1], which makes the math easy.
    double box_area = detection.bounding_box().width() * detection.bounding_box().height();
    double aspect_ratio = detection.bounding_box().width() / detection.bounding_box().height();

    // If there are many labels for a detection, we don't want to send metrics for all of them, as
    // the metrics will not be useful after some point. Instead, only send metrics for the top K
    // classes.
    for (const auto& label : detection.label()) {
      top_k_classes.push({label.label(), label.score()});
      if (top_k_classes.size() > kMaxK) {
        top_k_classes.pop();
      }
    }

    std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                       options.metric_attributes().end());

    for (size_t k = top_k_classes.size(); k > 0; --k) {
      const auto& [class_name, conf] = top_k_classes.top();

      attrs["class"] = class_name;
      attrs["k"] = std::to_string(k);

      confidence_hist_->Record(conf, attrs, {});

      top_k_classes.pop();
    }

    // Keep this after the loop so that attrs["class"] is set to top class.
    DCHECK(attrs["k"] == "1");
    attrs.erase("k");
    box_area_hist_->Record(box_area, attrs, {});
    box_aspect_ratio_hist_->Record(aspect_ratio, attrs, {});
  }

  return Status::OK();
}

REGISTER_CALCULATOR(DetectionsMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
