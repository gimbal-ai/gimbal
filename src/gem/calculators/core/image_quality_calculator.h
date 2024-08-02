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

#pragma once
#include <cstdint>

#include <mediapipe/framework/calculator_base.h>
#include <opencv2/quality/qualitybrisque.hpp>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/status.h"
#include "src/common/metrics/metrics_system.h"

namespace gml::gem::calculators::core {

/**
 * ImageQualityCalculator Graph API:
 *
 *  Inputs:
 *    IMAGE_FRAME mediapipe::ImageFrame
 *  Outputs:
 *    IMAGE_HIST contain image quality metrics as a histogram
 *    IMAGE_QUALITY contain image quality metrics
 *
 */
class ImageQualityCalculator : public mediapipe::CalculatorBase {
 public:
  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(OpenImpl(cc));
    return absl::OkStatus();
  }

  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(ProcessImpl(cc));
    return absl::OkStatus();
  }

  absl::Status Close(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(CloseImpl(cc));
    return absl::OkStatus();
  }

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc);
  Status ProcessImpl(mediapipe::CalculatorContext* cc);
  Status CloseImpl(mediapipe::CalculatorContext* cc);

 private:
  cv::Ptr<cv::quality::QualityBRISQUE> brisque_calc;
  opentelemetry::metrics::Gauge<double>* brisque_score_;
  opentelemetry::metrics::Gauge<double>* blurriness_score_;

  internal::api::core::v1::ImageQualityMetrics metrics_;
  int64_t frame_count = 0;
};

}  // namespace gml::gem::calculators::core
