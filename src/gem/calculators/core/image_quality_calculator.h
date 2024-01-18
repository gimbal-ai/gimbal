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
  std::unique_ptr<opentelemetry::metrics::Gauge<double>> brisque_score_;
  std::unique_ptr<opentelemetry::metrics::Gauge<double>> blurriness_score_;

  internal::api::core::v1::ImageQualityMetrics metrics_;
  int64_t frame_count = 0;
};

}  // namespace gml::gem::calculators::core
