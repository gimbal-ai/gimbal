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

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <absl/strings/str_format.h>

// Ordering of these includes is sensitive :(
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <NvEglRenderer.h>
#include "NvBufSurface.h"

#include "argus_cam.h"

constexpr int kNumFrames = 100;
constexpr int kDeviceNum = 0;

int main(int argc, char** argv) {
  gml::EnvironmentGuard env_guard(&argc, argv);

  gml::ArgusCam argus_cam;
  GML_EXIT_IF_ERROR(argus_cam.Init(kDeviceNum));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto renderer =
      std::unique_ptr<NvEglRenderer>(NvEglRenderer::createEglRenderer("renderer0", 640, 480, 0, 0));

  for (int i = 0; i < kNumFrames; ++i) {
    printf("--- %d\n", i);
    GML_ASSIGN_OR_EXIT(std::unique_ptr<gml::NvBufSurfaceWrapper> image_buf_fd,
                       argus_cam.ConsumeFrame());

    int buf_fd = image_buf_fd->fd();
    CHECK(buf_fd > 0) << "FD not valid.";

    renderer->render(buf_fd);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double elapsed_time_seconds =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) /
      1000;
  double effective_fps = static_cast<double>(kNumFrames) / elapsed_time_seconds;

  LOG(INFO) << absl::StrFormat("Elapsed time [seconds] = %g\n", elapsed_time_seconds);
  LOG(INFO) << absl::StrFormat("Frames captured = %d\n", kNumFrames);
  LOG(INFO) << absl::StrFormat("Effective FPS = %g\n", effective_fps);

  argus_cam.Stop();

  return 0;
}
