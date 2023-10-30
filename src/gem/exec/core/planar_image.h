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
#include "src/common/base/error.h"

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
 *
 * Implementations of this interface are under PlanarImageFor<TImage>. See that class for how to
 * implement a new image format.
 */
class PlanarImage {
 public:
  struct Plane {
    Plane() = default;
    Plane(const uint8_t* data, int row_stride) : data(data), row_stride(row_stride), bytes(0) {}
    const uint8_t* data;
    // Planes can have padding at the end of the row, so row stride specifies the distance in bytes
    // from a row to the next row. row_stride = ceil(width*bits/8) + padding_bytes.
    int row_stride;

    // Number of bytes in the plane. This is set based on the format so template specializations of
    // `PlanarImageFor` don't need to set it.
    size_t bytes;
  };

  virtual ~PlanarImage() {}

  virtual size_t Width() const = 0;
  virtual size_t Height() const = 0;
  virtual ImageFormat Format() const = 0;

  virtual const std::vector<Plane>& Planes() const = 0;
  virtual std::vector<Plane>& Planes() = 0;
};

/**
 * To implement the interface for an image format, provide implementations for the following
 * three functions in a cc file:
 *   template <>
 *   Status PlanarImageFor<TYourImageType>::BuildPlanes()
 *
 *   template<>
 *   size_t PlanarImageFor<TYourImageType>::Width() const
 *
 *   template<>
 *   size_t PlanarImageFor<TYourImageType>::Height() const
 */
template <typename TImage>
class PlanarImageFor : public PlanarImage {
 public:
  static StatusOr<std::unique_ptr<PlanarImage>> Create(std::unique_ptr<TImage> image,
                                                       ImageFormat format) {
    auto planar_image = absl::WrapUnique(new PlanarImageFor<TImage>(std::move(image)));
    GML_RETURN_IF_ERROR(planar_image->BuildPlanes());
    planar_image->format_ = format;
    GML_RETURN_IF_ERROR(planar_image->PlaneSizesFromFormat());
    return absl::WrapUnique<PlanarImage>(planar_image.release());
  }

  const std::vector<Plane>& Planes() const override { return planes_; }
  std::vector<Plane>& Planes() override { return planes_; }

  // Width() and Height() need to be implemented with template specialization.
  size_t Width() const override;
  size_t Height() const override;

  ImageFormat Format() const override { return format_; }

 private:
  // BuildPlanes() needs to be implemented with template specialization.
  Status BuildPlanes();

  Status PlaneSizesFromFormat() {
    switch (Format()) {
      case ImageFormat::YUV_I420:
        if (planes_.size() != 3) {
          return error::Internal(
              "PlanarImageFor<$0> created $1 planes instead of 3 for YUV_I420 "
              "format",
              typeid(TImage).name(), planes_.size());
        }
        planes_[0].bytes = planes_[0].row_stride * Height();
        planes_[1].bytes = planes_[1].row_stride * ((Height() + 1) / 2);
        planes_[2].bytes = planes_[1].bytes;
        return Status::OK();
      default:
        return error::InvalidArgument("Unknown ImageFormat: $0", magic_enum::enum_name(Format()));
    }
  }

  PlanarImageFor(std::unique_ptr<TImage> image) : image_(std::move(image)) {}

  std::unique_ptr<TImage> image_;
  ImageFormat format_;
  std::vector<PlanarImage::Plane> planes_;
};

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
