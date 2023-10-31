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

namespace gml {
namespace gem {
namespace testing {

cv::Mat LoadTestImageAsOpencvMat();

void LoadTestImageAsImageFrame(mediapipe::ImageFrame* image_frame);

void LoadTestImageAsYUVImage(mediapipe::YUVImage* yuv_image);

}  // namespace testing
}  // namespace gem
}  // namespace gml
