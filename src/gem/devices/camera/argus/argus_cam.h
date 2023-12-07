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

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

namespace gml::gem::devices::argus {

/**
 * Provides a simple access model to the Argus camera on Nvidia Jetson devices.
 */
class ArgusCam {
  static constexpr uint64_t kDefaultTargetFrameRate = 30;

 public:
  explicit ArgusCam(uint64_t target_frame_rate = kDefaultTargetFrameRate)
      : target_frame_rate_(target_frame_rate) {}

  ~ArgusCam() { Stop(); }

  Status Init(std::string device_uuid);

  /**
   * Acquire a frame from the camera.
   * Essentially returns a FD to a buffer containing the image.
   */
  StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> ConsumeFrame();

  void Stop();

 private:
  Status SetupCamera(std::string device_uuid);
  StatusOr<Argus::SensorMode*> SelectSensorMode();
  Status CreateOutputStream();
  StatusOr<Argus::UniqueObj<Argus::Request>> PrepareRequest(Argus::SensorMode* sensor_mode,
                                                            Argus::OutputStream* output_stream);
  Status PrepareConsumer(Argus::OutputStream* output_stream);
  Status StartCapture(Argus::Request* request);

  uint64_t target_frame_rate_;
  Argus::UniqueObj<Argus::CameraProvider> camera_provider_obj_;
  Argus::CameraDevice* camera_device_ = nullptr;
  Argus::UniqueObj<Argus::CaptureSession> capture_session_obj_;
  Argus::UniqueObj<Argus::OutputStream> output_stream_obj_;
  Argus::UniqueObj<EGLStream::FrameConsumer> frame_consumer_obj_;
};

}  // namespace gml::gem::devices::argus
