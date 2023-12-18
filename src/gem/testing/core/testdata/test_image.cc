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

#include "src/gem/testing/core/testdata/test_image.h"

#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/formats/yuv_image.h>
#include <mediapipe/util/image_frame_util.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "src/common/base/base.h"
#include "src/common/bazel/runfiles.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace gml::gem::testing {

constexpr std::string_view kTestPNGPath = "src/gem/testing/core/testdata/test.jpg";
constexpr std::string_view kTestBarcodeJPGPath = "src/gem/testing/core/testdata/barcode.jpg";

namespace {
std::filesystem::path GetPathFromType(TestImageType type) {
  std::filesystem::path rel_path;
  switch (type) {
    case TestImageType::WITH_BARCODE:
      rel_path = kTestBarcodeJPGPath;
      break;
    default:
      rel_path = kTestPNGPath;
      break;
  }
  return bazel::RunfilePath(rel_path);
}
}  // namespace

cv::Mat LoadTestImageAsOpencvMat(TestImageType type) {
  auto test_path = GetPathFromType(type);
  auto mat = cv::imread(test_path.string());
  CHECK(!mat.empty());
  return mat;
}

void LoadTestImageAsImageFrame(mediapipe::ImageFrame* image_frame, TestImageType type) {
  auto mat = LoadTestImageAsOpencvMat(type);
  image_frame->Reset(mediapipe::ImageFormat::FORMAT_SRGB, mat.cols, mat.rows,
                     mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  CHECK(mat.type() == CV_8UC3);

  mat.copyTo(mediapipe::formats::MatView(image_frame));
}

void LoadTestImageAsYUVImage(mediapipe::YUVImage* yuv_image, TestImageType type) {
  mediapipe::ImageFrame image_frame;
  LoadTestImageAsImageFrame(&image_frame, type);
  mediapipe::image_frame_util::ImageFrameToYUVImage(image_frame, yuv_image);
}

}  // namespace gml::gem::testing
