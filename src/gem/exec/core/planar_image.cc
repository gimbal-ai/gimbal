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

// Interface implementation for mediapipe::YUVImage

template <>
Status PlanarImageFor<mediapipe::YUVImage>::BuildPlanes() {
  constexpr int kYUVImageMaxNumFrames = 3;

  mediapipe::YUVImage* image = image_.get();

  for (int i = 0; i < kYUVImageMaxNumFrames; ++i) {
    auto data = image->data(i);
    if (data == nullptr) {
      break;
    }
    auto stride = image->stride(i);
    planes_.emplace_back(data, stride);
  }
  return Status::OK();
}

template <>
size_t PlanarImageFor<mediapipe::YUVImage>::Width() const {
  return image_->width();
}

template <>
size_t PlanarImageFor<mediapipe::YUVImage>::Height() const {
  return image_->height();
}

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
