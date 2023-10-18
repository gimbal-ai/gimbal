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

#include "NvBufSurface.h"

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include "src/common/base/base.h"

namespace gml {

// A wrapper around the image buf fd, with managed resources.
// TODO(oazizi): Make this non-copyable.
class NvBufSurfaceWrapper {
 public:
  NvBufSurfaceWrapper(int fd) : fd_(fd) {}
  ~NvBufSurfaceWrapper() {
    if (fd_ != -1) {
      NvBufSurf::NvDestroy(fd_);
    }
  }
  int fd() { return fd_; }

 private:
  const int fd_ = -1;
};

/**
 * Provides a simple access model to the Argus camera on Nvidia Jetson devices.
 */
class ArgusCam {
 public:
  /**
   * Initialize the capture device. By default uses the first camera, but can choose other devices
   * as well.
   */
  Status Init(int device_num = 0);

  /**
   * Acquire a frame from the camera.
   * Essentially returns a FD to a buffer containing the image.
   */
  StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> ConsumeFrame();

  void Stop();

 private:
  Status SetupCamera(int device_num);
  StatusOr<Argus::SensorMode*> SelectSensorMode();
  Status CreateOutputStream();
  StatusOr<Argus::UniqueObj<Argus::Request>> PrepareRequest(Argus::SensorMode* sensor_mode,
                                                            Argus::OutputStream* output_stream);
  Status PrepareConsumer(Argus::OutputStream* output_stream);
  Status StartCapture(Argus::Request* request);

  Argus::UniqueObj<Argus::CameraProvider> camera_provider_obj_;
  Argus::CameraDevice* camera_device_ = nullptr;
  Argus::UniqueObj<Argus::CaptureSession> capture_session_obj_;
  Argus::UniqueObj<Argus::OutputStream> output_stream_obj_;
  Argus::UniqueObj<EGLStream::FrameConsumer> frame_consumer_obj_;
};

}  // namespace gml