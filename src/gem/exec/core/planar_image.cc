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

#include <mediapipe/framework/formats/yuv_image.h>

#include "src/common/base/base.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {
constexpr int kYUVImageMaxNumFrames = 3;

template <>
StatusOr<std::vector<PlanarImage::Plane>> PlanesFromImage<mediapipe::YUVImage>(
    mediapipe::YUVImage* image) {
  std::vector<PlanarImage::Plane> planes;
  for (int i = 0; i < kYUVImageMaxNumFrames; ++i) {
    auto data = image->mutable_data(i);
    if (data == nullptr) {
      break;
    }
    auto stride = image->stride(i);
    planes.push_back(PlanarImage::Plane{
        data,
        stride,
    });
  }
  return planes;
}

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
