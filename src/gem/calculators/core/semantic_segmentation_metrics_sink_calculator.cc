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

#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/core/optionspb/semantic_segmentation_metrics_sink_calculator_options.pb.h"

namespace gml::gem::calculators::core {
using ::gml::internal::api::core::v1::Segmentation;

/**
 *  SemanticSegmentationMetricsSinkCalculator Graph API:
 *
 *  Inputs:
 *    internal::api::corepb::v1::Segmentation a semantic segmentation proto.
 *
 *  No real outputs, outputs stats to opentelemetry.
 */

class SemanticSegmentationMetricsSinkCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);

  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  opentelemetry::metrics::Histogram<double>* area_percentage_hist_;
};

constexpr std::string_view kFinishedTag = "FINISHED";

// Area percentage histogram bounds (0% to 100% in 5% increments)
const std::vector<double> kAreaPercentageBounds = []() {
  std::vector<double> temp;
  for (int i = 0; i <= 100; i += 5) {
    temp.push_back(i / 100.0);
  }
  return temp;
}();

absl::Status SemanticSegmentationMetricsSinkCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<Segmentation>();
  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs().Tag(kFinishedTag).Set<bool>();
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status SemanticSegmentationMetricsSinkCalculator::Open(mediapipe::CalculatorContext*) {
  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  area_percentage_hist_ = metrics_system.GetOrCreateHistogramWithBounds<double>(
      "gml_gem_pipe_segmentation_area_percentage",
      "Percentage of area used per class in segmentation mask.", kAreaPercentageBounds);
  return absl::OkStatus();
}

absl::Status SemanticSegmentationMetricsSinkCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& options = cc->Options<optionspb::SemanticSegmentationMetricsSinkCalculatorOptions>();
  auto& segmentation = cc->Inputs().Index(0).Get<Segmentation>();

  int64_t total_pixels = segmentation.width() * segmentation.height();
  std::unordered_map<std::string, int64_t> class_pixel_counts;

  for (const auto& mask : segmentation.masks()) {
    int64_t pixel_count = 0;
    for (int i = 0; i < mask.run_length_encoding_size(); ++i) {
      if (i % 2 == 1) {
        pixel_count += mask.run_length_encoding(i);
      }
    }
    class_pixel_counts[mask.label()] = pixel_count;
  }

  std::unordered_map<std::string, std::string> attrs(options.metric_attributes().begin(),
                                                     options.metric_attributes().end());
  // Calculate and record area percentage for each class
  for (const auto& [class_name, pixel_count] : class_pixel_counts) {
    double area_percentage = static_cast<double>(pixel_count) / static_cast<double>(total_pixels);

    attrs["class"] = class_name;

    area_percentage_hist_->Record(area_percentage, attrs, {});
  }

  if (cc->Outputs().HasTag(kFinishedTag)) {
    cc->Outputs()
        .Tag(kFinishedTag)
        .AddPacket(mediapipe::MakePacket<bool>(true).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

REGISTER_CALCULATOR(SemanticSegmentationMetricsSinkCalculator);

}  // namespace gml::gem::calculators::core
