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

#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/yuv_image.h>
#include <opencv2/core.hpp>

namespace gml::gem::testing {

enum class TestImageType {
  DEFAULT = 0,
  WITH_BARCODE = 1,
};

cv::Mat LoadTestImageAsOpencvMat(TestImageType type = TestImageType::DEFAULT);

void LoadTestImageAsImageFrame(mediapipe::ImageFrame* image_frame,
                               TestImageType type = TestImageType::DEFAULT);

void LoadTestImageAsYUVImage(mediapipe::YUVImage* yuv_image,
                             TestImageType type = TestImageType::DEFAULT);

}  // namespace gml::gem::testing
