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

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <memory>

namespace gml::gem::calculators::ffmpeg {

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

}  // namespace gml::gem::calculators::ffmpeg
