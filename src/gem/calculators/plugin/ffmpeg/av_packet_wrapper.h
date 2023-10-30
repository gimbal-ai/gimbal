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

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace gml {
namespace gem {
namespace calculators {
namespace ffmpeg {

class AVPacketWrapper {
 public:
  static std::unique_ptr<AVPacketWrapper> Create() {
    return std::unique_ptr<AVPacketWrapper>(new AVPacketWrapper);
  }
  static std::unique_ptr<AVPacketWrapper> CreateRef(AVPacket* packet) {
    return std::unique_ptr<AVPacketWrapper>(new AVPacketWrapper(packet));
  }
  ~AVPacketWrapper() {
    if (ref_) {
      av_packet_unref(packet_);
    } else {
      av_packet_free(&packet_);
    }
  }

  AVPacket* mutable_packet() { return packet_; }

  const AVPacket* packet() const { return packet_; }

 protected:
  AVPacketWrapper() : ref_(false) { packet_ = av_packet_alloc(); }
  explicit AVPacketWrapper(AVPacket* packet) : AVPacketWrapper() {
    av_packet_ref(packet_, packet);
    ref_ = true;
  }

 private:
  AVPacket* packet_;
  bool ref_;
};

}  // namespace ffmpeg
}  // namespace calculators
}  // namespace gem
}  // namespace gml
