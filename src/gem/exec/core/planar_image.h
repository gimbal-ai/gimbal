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

#include <mediapipe/framework/formats/yuv_image.h>

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {

/**
 * ImageFormat enumerates our names for image formats. Since there are many names for image formats
 * (often multiple names for the same format), we should standardize on a single name for any given
 * arrangement of image data.
 */
enum class ImageFormat {
  UNKNOWN = 0,
  // 3 planes. 8-bit Y plane, followed by 8 bit 2x2 subsampled U and V planes.
  YUV_I420 = 1,
};

/**
 * PlanarImage is the interface for accessing image data by plane.
 *
 * PlanarImage is agnostic to the underlying image format. Packed image formats can be represented
 * by using a single plane.
 *
 * Accessing the data in the planes is also format dependent. Some formats don't have planes of the
 * same size, so `Width()` and `Height()` represent the width and height of the image itself and
 * don't necessarily correspond to the width and height of the planes.
 */
class PlanarImage {
 public:
  struct Plane {
    uint8_t* data;
    // Planes can have padding at the end of the row, so row stride specifies the distance in bytes
    // from a row to the next row. row_stride = ceil(width*bits/8) + padding_bytes.
    int row_stride;
  };

  virtual ~PlanarImage() {}

  virtual size_t Width() const = 0;
  virtual size_t Height() const = 0;
  virtual ImageFormat Format() const = 0;

  virtual const std::vector<Plane>& Planes() const = 0;
  virtual std::vector<Plane>& Planes() = 0;
};

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
