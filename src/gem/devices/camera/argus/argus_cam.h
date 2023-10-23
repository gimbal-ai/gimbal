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
class NvBufSurfaceWrapper : public NotCopyable {
 public:
  static StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> Create(int fd) {
    NvBufSurface* nvbuf_surf = nullptr;
    int retval = NvBufSurfaceFromFd(fd, reinterpret_cast<void**>(&nvbuf_surf));
    if (retval != 0) {
      return error::Internal("NvBufSurfaceFromFd failed.");
    }

    if (nvbuf_surf->batchSize < 1) {
      return error::Internal("No image batches in buffer.");
    }

    if (nvbuf_surf->batchSize > 1) {
      LOG(WARNING) << absl::Substitute(
          "Expected 1 batch in buffer, but found $0. Using buffer [0].", nvbuf_surf->batchSize);
    }

    return absl::WrapUnique<NvBufSurfaceWrapper>(new NvBufSurfaceWrapper(fd, nvbuf_surf));
  }

  ~NvBufSurfaceWrapper() {
    if (fd_ != -1) {
      NvBufSurf::NvDestroy(fd_);
    }
  }

  void DumpInfo() {
    LOG(INFO) << absl::Substitute("nvbuf_surf: batchSize $0", nvbuf_surf_->batchSize);
    LOG(INFO) << absl::Substitute("nvbuf_surf: numFilled $0", nvbuf_surf_->numFilled);

    // Only supporting the first buffer for now.
    const NvBufSurfaceParams& surf_params = nvbuf_surf_->surfaceList[0];

    LOG(INFO) << absl::Substitute(
        "surf_params[0]: WxH=$0x$1 pitch=$2 color_format=$3 layout=$4 dataSize=$5 num_planes=$6",
        surf_params.width, surf_params.height, surf_params.pitch, surf_params.colorFormat,
        surf_params.layout, surf_params.dataSize, surf_params.planeParams.num_planes);
    for (uint32_t i = 0; i < surf_params.planeParams.num_planes; ++i) {
      const auto& plane_params = surf_params.planeParams;
      LOG(INFO) << absl::Substitute(
          "plane_params[$0] WxH=$1x$2 pitch=$3 offset=$4 psize=$5 bytesPerPix=$6", i,
          plane_params.width[i], plane_params.height[i], plane_params.pitch[i],
          plane_params.offset[i], plane_params.psize[i], plane_params.bytesPerPix[i]);
    }
  }

  /**
   * Maps the buffer for CPU access.
   * Information about where the different planes are located are populated in the internal
   * NvBufSurface data structure, which can be accessed via surface().mappedAddr.
   */
  Status MapForCpu() {
    const int kAllBuffers = -1;
    const int kAllPlanes = -1;

    int retval = NvBufSurfaceMap(nvbuf_surf_, kAllBuffers, kAllPlanes, NVBUF_MAP_READ_WRITE);
    if (retval < 0) {
      return error::Internal("NvBufSurfaceMap failed.");
    }

    NvBufSurfaceSyncForCpu(nvbuf_surf_, kAllBuffers, kAllPlanes);

    return Status::OK();
  }

  int fd() { return fd_; }
  const NvBufSurfaceParams& surface() { return nvbuf_surf_->surfaceList[0]; }

 private:
  NvBufSurfaceWrapper(int fd, NvBufSurface* nvbuf_surf) : fd_(fd), nvbuf_surf_(nvbuf_surf) {}

  const int fd_ = -1;
  NvBufSurface* nvbuf_surf_ = nullptr;
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
