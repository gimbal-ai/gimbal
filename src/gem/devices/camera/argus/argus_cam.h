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

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

namespace gml::gem::devices::argus {

/**
 * Provides a simple access model to the Argus camera on Nvidia Jetson devices.
 */
class ArgusCam {
 public:
  ~ArgusCam() { Stop(); }

  Status Init();

  /**
   * Acquire a frame from the camera.
   * Essentially returns a FD to a buffer containing the image.
   */
  StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> ConsumeFrame();

  /*
   * Returns the timestamp of the last frame capture in nanoseconds.
   * This is expected to be called after ConsumeFrame.
   */
  uint64_t GetLastCaptureNS() { return last_capture_ns_; }

  void Stop();

  const std::string& UUID() const { return uuid_; }

 private:
  // Hide the constructor so that only ArgusCamFactory can create ArgusCam objects.
  ArgusCam(Argus::CameraDevice* device, Argus::UniqueObj<Argus::CaptureSession> session,
           uint64_t target_frame_rate, std::string uuid)
      : target_frame_rate_(target_frame_rate),
        camera_device_(device),
        capture_session_obj_(std::move(session)),
        uuid_(std::move(uuid)) {}
  friend class ArgusCamFactory;

  StatusOr<Argus::SensorMode*> SelectSensorMode();
  Status CreateOutputStream();
  StatusOr<Argus::UniqueObj<Argus::Request>> PrepareRequest(Argus::SensorMode* sensor_mode,
                                                            Argus::OutputStream* output_stream);
  Status PrepareConsumer(Argus::OutputStream* output_stream);
  Status StartCapture(Argus::Request* request);

  uint64_t last_capture_ns_ = 0;
  uint64_t target_frame_rate_;
  Argus::CameraDevice* camera_device_;
  Argus::UniqueObj<Argus::CaptureSession> capture_session_obj_;
  Argus::UniqueObj<Argus::OutputStream> output_stream_obj_;
  Argus::UniqueObj<EGLStream::FrameConsumer> frame_consumer_obj_;
  std::string uuid_;
};

}  // namespace gml::gem::devices::argus
