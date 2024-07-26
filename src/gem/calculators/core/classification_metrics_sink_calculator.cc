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

#include "src/gem/calculators/core/classification_metrics_sink_calculator.h"

#include <queue>

#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/classification_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Classification;

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

Status ClassificationMetricsSinkCalculator::BuildMetrics(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();

  confidence_hist_ = metrics_system.CreateHistogramWithBounds<double>(
      "gml_gem_model_classification_confidence", "Confidence scores of model predictions.",
      kConfidenceClassesBounds);
  return Status::OK();
}

Status ClassificationMetricsSinkCalculator::RecordMetrics(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::ClassificationMetricsSinkCalculatorOptions>();
  auto& classification = cc->Inputs().Index(0).Get<Classification>();

  auto cmp = [](const std::pair<std::string, double>& lhs,
                const std::pair<std::string, double>& rhs) { return lhs.second > rhs.second; };

  std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>,
                      decltype(cmp)>
      top_k_classes(cmp);

  // Sort the labels by score, keeping top K only.
  for (int i = 0; i < classification.label_size(); ++i) {
    const auto& label = classification.label(i);
    top_k_classes.push({label.label(), label.score()});
    if (top_k_classes.size() > kMaxK) {
      top_k_classes.pop();
    }
  }

  for (size_t k = top_k_classes.size(); k > 0; --k) {
    const auto& [class_name, conf] = top_k_classes.top();

    std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                       options.metric_attributes().end());
    attrs["class"] = class_name;
    attrs["k"] = std::to_string(k);

    confidence_hist_->Record(conf, attrs, {});

    top_k_classes.pop();
  }

  return Status::OK();
}

REGISTER_CALCULATOR(ClassificationMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
