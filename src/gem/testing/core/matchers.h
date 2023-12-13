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

#include <magic_enum.hpp>

#include "src/common/testing/testing.h"
#include "src/gem/exec/core/planar_image.h"

namespace gml::gem::testing {

MATCHER_P(PlanarImageEq, expected, "") {
  auto* actual = arg;
  if (actual->Width() != expected->Width()) {
    return false;
  }
  if (actual->Height() != expected->Height()) {
    return false;
  }

  if (actual->Format() != expected->Format()) {
    return false;
  }
  if (actual->Planes().size() != expected->Planes().size()) {
    return false;
  }
  for (const auto& [plane_idx, actual_plane] : Enumerate(actual->Planes())) {
    const auto& expected_plane = expected->Planes()[plane_idx];
    if (actual_plane.row_stride != expected_plane.row_stride) {
      return false;
    }
    if (actual_plane.bytes != expected_plane.bytes) {
      return false;
    }

    size_t width = actual->Width();
    size_t height = actual->Height();
    size_t stride = actual_plane.row_stride;
    switch (expected->Format()) {
      case exec::core::ImageFormat::YUV_I420:
        if (plane_idx != 0) {
          width = (width + 1) / 2;
          height = (height + 1) / 2;
        }
        break;
      default:
        CHECK(false) << "Unimplemented ImageFormat in PlanarImageEq matcher: "
                     << magic_enum::enum_name<exec::core::ImageFormat>(expected->Format());
    }
    for (size_t row_idx = 0; row_idx < height; ++row_idx) {
      const auto* actual_row = &actual_plane.data[row_idx * stride];
      const auto* expected_row = &actual_plane.data[row_idx * stride];
      if (std::memcmp(actual_row, expected_row, width) != 0) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace gml::gem::testing
