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

#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/formats/yuv_image.h>
#include <mediapipe/util/image_frame_util.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tools/cpp/runfiles/runfiles.h"

#include "src/common/base/base.h"
#include "src/common/testing/test_environment.h"

namespace gml::gem::testing {

constexpr std::string_view kTestPNGPath = "src/gem/testing/core/testdata/test.jpg";

cv::Mat LoadTestImageAsOpencvMat() {
  auto test_path = ::gml::testing::BazelRunfilePath(std::filesystem::path(kTestPNGPath));
  auto mat = cv::imread(test_path.string());
  CHECK(!mat.empty());
  return mat;
}

void LoadTestImageAsImageFrame(mediapipe::ImageFrame* image_frame) {
  auto mat = LoadTestImageAsOpencvMat();
  image_frame->Reset(mediapipe::ImageFormat::SRGB, mat.cols, mat.rows,
                     mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  CHECK(mat.type() == CV_8UC3);

  mat.copyTo(mediapipe::formats::MatView(image_frame));
}

void LoadTestImageAsYUVImage(mediapipe::YUVImage* yuv_image) {
  mediapipe::ImageFrame image_frame;
  LoadTestImageAsImageFrame(&image_frame);
  mediapipe::image_frame_util::ImageFrameToYUVImage(image_frame, yuv_image);
}

}  // namespace gml::gem::testing
