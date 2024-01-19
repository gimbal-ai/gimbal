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

#include "src/gem/calculators/core/image_quality_calculator.h"

#include <limits>

#include <mediapipe/framework/calculator_registry.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/metrics/metrics_system.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kImageFrameTag = "IMAGE_FRAME";
constexpr std::string_view kHistOutputTag = "IMAGE_HIST";
constexpr std::string_view kQualityOutputTag = "IMAGE_QUALITY";

// The value of ~150.0 gives a reasonable threshold to say the image is not blurry.
constexpr double kVarianceScaleFactor = 50.0f;
constexpr int kHistBuckets = 64;
constexpr double kBlurrinessThreshold = 0.25f;
constexpr int64_t kBrisqueFramesToSkip = 15;

using internal::api::core::v1::ImageHistogram;
using internal::api::core::v1::ImageHistogramBatch;
using internal::api::core::v1::ImageQualityMetrics;

namespace {

struct HistComputeData {
  uchar min = std::numeric_limits<uchar>::max();
  uchar max = std::numeric_limits<uchar>::min();

  int sum = 0;
  std::vector<int> buckets;

  HistComputeData() : buckets(kHistBuckets) {}
};

void packIntoHistogramProto(ImageHistogram* hist, int size, const HistComputeData& data) {
  auto buckets = hist->mutable_bucket();
  for (int i : data.buckets) {
    buckets->Add(i);
  }

  hist->set_max(static_cast<float>(data.max) / 255.0f);
  hist->set_min(static_cast<float>(data.min) / 255.0f);
  hist->set_sum(static_cast<float>(data.sum) / 255.0f);
  hist->set_num(size);
}

void computeColorHistograms(ImageHistogramBatch* out, const cv::Mat& img) {
  CHECK(img.channels() == 3);
  std::vector<HistComputeData> hist_data(3);  // One per color channel.

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
      for (int c = 0; c < 3; c++) {
        hist_data[c].min = std::min(hist_data[c].min, pixel[c]);
        hist_data[c].max = std::max(hist_data[c].max, pixel[c]);

        hist_data[c].sum += pixel[c];

        int bucket = static_cast<int>(kHistBuckets * static_cast<float>(pixel[c]) / 256.0f);
        hist_data[c].buckets[bucket]++;
      }
    }
  }

  // Pack them into HistogramBatch.
  int size = img.rows * img.cols;
  auto rhist = out->add_histograms();
  rhist->set_channel(internal::api::core::v1::IMAGE_COLOR_CHANNEL_RED);
  packIntoHistogramProto(rhist, size, hist_data[0]);

  auto ghist = out->add_histograms();
  ghist->set_channel(internal::api::core::v1::IMAGE_COLOR_CHANNEL_GREEN);
  packIntoHistogramProto(ghist, size, hist_data[1]);

  auto bhist = out->add_histograms();
  bhist->set_channel(internal::api::core::v1::IMAGE_COLOR_CHANNEL_BLUE);
  packIntoHistogramProto(bhist, size, hist_data[2]);
}

}  // namespace

absl::Status ImageQualityCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  cc->Outputs().Tag(kHistOutputTag).Set<gml::internal::api::core::v1::ImageHistogramBatch>();
  cc->Outputs().Tag(kQualityOutputTag).Set<gml::internal::api::core::v1::ImageQualityMetrics>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

Status ImageQualityCalculator::OpenImpl(mediapipe::CalculatorContext*) {
  auto model_file = bazel::RunfilePath(
      "external/com_github_opencv_contrib/modules/quality/samples/brisque_model_live.yml");
  auto range_file = bazel::RunfilePath(
      "external/com_github_opencv_contrib/modules/quality/samples/brisque_range_live.yml");
  brisque_calc = cv::quality::QualityBRISQUE::create(model_file, range_file);

  auto& metrics_system = metrics::MetricsSystem::GetInstance();
  auto gml_meter = metrics_system.GetMeterProvider()->GetMeter("gml");
  brisque_score_ = gml_meter->CreateDoubleGauge("gml.gem.image_quality.brisque_score");
  blurriness_score_ = gml_meter->CreateDoubleGauge("gml.gem.image_quality.blurriness_score");

  return Status::OK();
}

Status ImageQualityCalculator::ProcessImpl(mediapipe::CalculatorContext* cc) {
  DEFER(frame_count++);
  const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();
  cv::Mat input_mat = mediapipe::formats::MatView(&image_frame);

  if (input_mat.channels() != 3) {
    return error::InvalidArgument("Only RGB images are supported. Image with $0 channels passed in",
                                  input_mat.channels());
  }

  cv::Mat gray_img, laplacian_img;
  // Convert the incoming RGB image into a grayscale image.
  // TODO(zasgar): Use ColorConvertCalculator so that it's usable by other
  // calculators.
  cv::cvtColor(input_mat, gray_img, cv::COLOR_RGB2GRAY);

  // The compute the blurriness we use the variance of the laplacian.
  cv::Scalar l_mean, l_stddev;
  cv::Laplacian(gray_img, laplacian_img, CV_8U);
  cv::meanStdDev(laplacian_img, l_mean, l_stddev);
  double var = l_stddev.val[0] * l_stddev.val[0];
  double norm = std::clamp((var / kVarianceScaleFactor), 0.0, 1.0);
  double blurriness = 1.0 - norm;
  double prev_blurriness = metrics_.blurriness_score();
  metrics_.set_blurriness_score(1.0 - norm);

  // Brisque is slow so we only compute it every few iterations or if there is a significant change
  // in the blurriness.
  if ((frame_count % kBrisqueFramesToSkip) == 0 ||
      std::abs(prev_blurriness - blurriness) > kBlurrinessThreshold) {
    cv::Scalar res = brisque_calc->compute(gray_img);
    metrics_.set_brisque_score(std::clamp((100 - res[0]) / 100.0, 0.0, 1.0));
  }

  // Record the otel metrics.
  brisque_score_->Record(metrics_.brisque_score());
  blurriness_score_->Record(metrics_.blurriness_score());

  // For the histograms we compute the R,G,B and grayscale histograms and store them
  // in the batch. We treat all pixels as normalized 0 - 1, although for computations we use the
  // underlying 8-bit representation for efficiency.
  ImageHistogramBatch hist_batch;
  computeColorHistograms(&hist_batch, input_mat);

  HistComputeData hist_data;
  for (int i = 0; i < gray_img.rows; ++i) {
    for (int j = 0; j < gray_img.cols; ++j) {
      uchar val = gray_img.at<uchar>(i, j);

      hist_data.min = std::min(hist_data.min, val);
      hist_data.max = std::max(hist_data.max, val);

      hist_data.sum += val;

      int bucket = static_cast<int>(kHistBuckets * static_cast<float>(val) / 256.0f);
      hist_data.buckets[bucket]++;
    }
  }

  auto ghist = hist_batch.add_histograms();
  ghist->set_channel(internal::api::core::v1::IMAGE_COLOR_CHANNEL_GRAY);
  packIntoHistogramProto(ghist, gray_img.rows * gray_img.cols, hist_data);

  cc->Outputs()
      .Tag(kHistOutputTag)
      .AddPacket(mediapipe::MakePacket<ImageHistogramBatch>(std::move(hist_batch))
                     .At(cc->InputTimestamp()));

  cc->Outputs()
      .Tag(kQualityOutputTag)
      .AddPacket(mediapipe::MakePacket<ImageQualityMetrics>(metrics_).At(cc->InputTimestamp()));

  return Status::OK();
}

Status ImageQualityCalculator::CloseImpl(mediapipe::CalculatorContext*) { return Status::OK(); }

REGISTER_CALCULATOR(ImageQualityCalculator);

}  // namespace gml::gem::calculators::core
