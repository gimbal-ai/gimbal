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
