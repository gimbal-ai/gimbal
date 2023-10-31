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

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {

using ::gml::internal::api::core::v1::H264Chunk;
using ::gml::internal::api::core::v1::ImageOverlayChunk;

/**
 * ControlExecutionContext allows calculators to interface with GEM's controller.
 * */
class ControlExecutionContext : public ExecutionContext {
 public:
  using VideoWithOverlaysCallback =
      std::function<Status(const std::vector<ImageOverlayChunk>&, const std::vector<H264Chunk>&)>;

  void RegisterVideoWithOverlaysCallback(VideoWithOverlaysCallback cb) {
    video_w_overlays_cb_ = cb;
  }

  void ClearVideoWithOverlaysCallback() { video_w_overlays_cb_ = nullptr; }

  bool HasVideoWithOverlaysCallback() { return !!video_w_overlays_cb_; }

  const VideoWithOverlaysCallback& GetVideoWithOverlaysCallback() { return video_w_overlays_cb_; }

 private:
  VideoWithOverlaysCallback video_w_overlays_cb_;
};

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml