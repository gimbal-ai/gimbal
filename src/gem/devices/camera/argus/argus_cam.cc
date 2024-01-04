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

#include "src/gem/devices/camera/argus/argus_cam.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <NvBufSurface.h>
#include <sole.hpp>

#include "src/common/base/base.h"
#include "src/gem/devices/camera/argus/uuid_utils.h"

namespace gml::gem::devices::argus {

StatusOr<Argus::SensorMode*> ArgusCam::SelectSensorMode() {
  // Get the camera properties.
  Argus::ICameraProperties* camera_properties =
      Argus::interface_cast<Argus::ICameraProperties>(camera_device_);
  if (camera_properties == nullptr) {
    return error::Internal("Failed to get CameraProperties.");
  }

  // Get the sensor modes from the camer properties.
  std::vector<Argus::SensorMode*> sensor_modes;
  camera_properties->getBasicSensorModes(&sensor_modes);
  if (sensor_modes.size() == 0) {
    return error::Internal("Failed to get sensor modes.");
  }

  LOG(INFO) << "Available Sensor modes :";
  for (uint32_t i = 0; i < sensor_modes.size(); i++) {
    Argus::ISensorMode* sensor_mode = Argus::interface_cast<Argus::ISensorMode>(sensor_modes[i]);
    Argus::Size2D<uint32_t> resolution = sensor_mode->getResolution();
    LOG(INFO) << absl::Substitute("[$0] WxH = $1x$2", i, resolution.width(), resolution.height());
  }

  // TODO(oazizi): Make a better selection than a hard-coded value.
  const int kSensorMode = sensor_modes.size() - 1;
  LOG(INFO) << absl::Substitute("Using sensor mode: $0", kSensorMode);
  return sensor_modes[kSensorMode];
}

Status ArgusCam::CreateOutputStream() {
  // Unset the DISPLAY environment variable, which appears to affect libEGL, causing failures.
  unsetenv("DISPLAY");

  Argus::ICaptureSession* capture_session =
      Argus::interface_cast<Argus::ICaptureSession>(capture_session_obj_);
  if (capture_session == nullptr) {
    return error::Internal("Capture session is null.");
  }

  // Create output stream settings.
  Argus::UniqueObj<Argus::OutputStreamSettings> output_stream_settings_obj(
      capture_session->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
  Argus::IEGLOutputStreamSettings* output_stream_settings =
      Argus::interface_cast<Argus::IEGLOutputStreamSettings>(output_stream_settings_obj);
  if (output_stream_settings == nullptr) {
    return error::Internal("Failed to create OutputStreamSettings.");
  }

  // Set stream format and resolution.
  output_stream_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
  output_stream_settings->setResolution(Argus::Size2D<uint32_t>(1280, 720));
  output_stream_settings->setMetadataEnable(true);

  // Create the output stream.
  output_stream_obj_ = Argus::UniqueObj<Argus::OutputStream>(
      capture_session->createOutputStream(output_stream_settings_obj.get()));

  return Status::OK();
}

StatusOr<Argus::UniqueObj<Argus::Request>> ArgusCam::PrepareRequest(
    Argus::SensorMode* sensor_mode, Argus::OutputStream* output_stream) {
  Argus::Status s;

  Argus::ICaptureSession* capture_session =
      Argus::interface_cast<Argus::ICaptureSession>(capture_session_obj_);
  if (capture_session == nullptr) {
    return error::Internal("Capture session is null.");
  }

  Argus::UniqueObj<Argus::Request> request_obj =
      Argus::UniqueObj<Argus::Request>(capture_session->createRequest());
  Argus::IRequest* request = Argus::interface_cast<Argus::IRequest>(request_obj);
  if (!request) {
    return error::Internal("Failed to create Request .");
  }

  s = request->enableOutputStream(output_stream);
  if (s != Argus::STATUS_OK) {
    return error::Internal("Failed to enable OutputStream.");
  }

  Argus::ISourceSettings* source_settings =
      Argus::interface_cast<Argus::ISourceSettings>(request_obj);
  if (source_settings == nullptr) {
    return error::Internal("Failed to get SourceSettings.");
  }
  source_settings->setSensorMode(sensor_mode);
  source_settings->setFrameDurationRange(
      Argus::Range<uint64_t>(1000 * 1000 * 1000 / target_frame_rate_));

  return request_obj;
}

Status ArgusCam::PrepareConsumer(Argus::OutputStream* output_stream) {
  frame_consumer_obj_ =
      Argus::UniqueObj<EGLStream::FrameConsumer>(EGLStream::FrameConsumer::create(output_stream));

  return Status::OK();
}

Status ArgusCam::StartCapture(Argus::Request* request) {
  Argus::Status s;

  Argus::ICaptureSession* capture_session =
      Argus::interface_cast<Argus::ICaptureSession>(capture_session_obj_);
  if (capture_session == nullptr) {
    return error::Internal("Capture session is null.");
  }

  s = capture_session->repeat(request);
  if (s != Argus::STATUS_OK) {
    return error::Internal("Failed to submit capture request.");
  }

  return Status::OK();
}

Status ArgusCam::Init() {
  GML_ASSIGN_OR_RETURN(Argus::SensorMode * sensor_mode, SelectSensorMode());
  GML_RETURN_IF_ERROR(CreateOutputStream());
  GML_ASSIGN_OR_RETURN(Argus::UniqueObj<Argus::Request> request,
                       PrepareRequest(sensor_mode, output_stream_obj_.get()));
  GML_RETURN_IF_ERROR(PrepareConsumer(output_stream_obj_.get()));
  GML_RETURN_IF_ERROR(StartCapture(request.get()));
  return Status::OK();
}

StatusOr<std::unique_ptr<NvBufSurfaceWrapper>> ArgusCam::ConsumeFrame() {
  Argus::Status s;

  EGLStream::IFrameConsumer* frame_consumer =
      Argus::interface_cast<EGLStream::IFrameConsumer>(frame_consumer_obj_);
  if (frame_consumer == nullptr) {
    return error::Internal("Failed to initialize Consumer.");
  }

  // Get the resulting frame from the consumer.
  constexpr int kTimeout = 1000000000;
  Argus::UniqueObj<EGLStream::Frame> frame_obj(frame_consumer->acquireFrame(kTimeout, &s));
  EGLStream::IFrame* frame = Argus::interface_cast<EGLStream::IFrame>(frame_obj);
  if (frame == nullptr) {
    return error::Internal("Failed to get Frame");
  }

  EGLStream::Image* image = frame->getImage();
  last_capture_ns_ = frame->getTime();

  // Metadata is also available, if desired.
  VLOG(2) << absl::Substitute("Frame num = $0", frame->getNumber());
  VLOG(2) << absl::Substitute("Frame time = $0", frame->getTime());

  Argus::IEGLOutputStream* output_stream =
      Argus::interface_cast<Argus::IEGLOutputStream>(output_stream_obj_);
  if (output_stream == nullptr) {
    return error::Internal("Failed to get OutputStream");
  }

  // Copy to NV native buffer.
  EGLStream::NV::IImageNativeBuffer* nv_buffer =
      Argus::interface_cast<EGLStream::NV::IImageNativeBuffer>(image);
  if (nv_buffer == nullptr) {
    return error::Internal("Failed to get nv_buffer.");
  }

  int image_buf_fd;
  image_buf_fd = nv_buffer->createNvBuffer(output_stream->getResolution(),
                                           NVBUF_COLOR_FORMAT_YUV420, NVBUF_LAYOUT_PITCH);

  NvBufSurface* nvbuf_surf = nullptr;
  int retval = NvBufSurfaceFromFd(image_buf_fd, reinterpret_cast<void**>(&nvbuf_surf));
  if (retval != 0) {
    return error::Internal("NvBufSurfaceFromFd() failed.");
  }

  if (nvbuf_surf->batchSize < 1) {
    return error::Internal("No image batches in buffer.");
  }

  if (nvbuf_surf->batchSize > 1) {
    LOG(WARNING) << absl::Substitute("Expected 1 batch in buffer, but found $0. Using buffer [0].",
                                     nvbuf_surf->batchSize);
  }

  GML_ASSIGN_OR_RETURN(std::unique_ptr<NvBufSurfaceWrapper> nvbuf_surf_wrapper,
                       NvBufSurfaceWrapper::Create(nvbuf_surf));

  // Dump information on the first frame only, to avoid being noisy.
  // TODO(oazizi): Change this to a VLOG once we have more confidence.
  if (frame->getNumber() == 0) {
    nvbuf_surf_wrapper->DumpInfo();
  }

  // TODO(oazizi): May want to find a way to recycle buffers. Needs perf study.
  //               See code below, for alternate call if buffer recycling is available.
  //  s = nv_buffer->copyToNvBuffer(src->frameInfo->fd)
  //  if (s != Argus::STATUS_OK) {
  //    return absl::Internal("Failed to copy to NV buffer.");
  //  }

  return nvbuf_surf_wrapper;
}

void ArgusCam::Stop() {
  Argus::ICaptureSession* capture_session =
      Argus::interface_cast<Argus::ICaptureSession>(capture_session_obj_);
  if (capture_session != nullptr) {
    capture_session->stopRepeat();
    capture_session->waitForIdle();
  }

  Argus::IEGLOutputStream* output_stream =
      Argus::interface_cast<Argus::IEGLOutputStream>(output_stream_obj_);
  if (output_stream != nullptr) {
    output_stream->disconnect();
  }
}

}  // namespace gml::gem::devices::argus
