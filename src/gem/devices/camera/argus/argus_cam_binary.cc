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

#include <nvbufsurface.h>

// This define does not work in our current build system
// because of v4l header conflicts.
// #define NV_EGL_RENDERER

#ifdef NV_EGL_RENDERER
#include <NvEglRenderer.h>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

#include "argus_cam.h"

constexpr int kNumFrames = 100;
constexpr int kDeviceNum = 0;

int main(int argc, char** argv) {
  gml::EnvironmentGuard env_guard(&argc, argv);

  using gml::gem::devices::argus::ArgusCam;
  using gml::gem::devices::argus::NvBufSurfaceWrapper;

  ArgusCam argus_cam;
  GML_EXIT_IF_ERROR(argus_cam.Init(kDeviceNum));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#ifdef NV_EGL_RENDERER
  auto renderer =
      std::unique_ptr<NvEglRenderer>(NvEglRenderer::createEglRenderer("renderer0", 640, 480, 0, 0));
#endif

  for (int i = 0; i < kNumFrames; ++i) {
    LOG(INFO) << absl::Substitute("--- $0", i);
    GML_ASSIGN_OR_EXIT(std::unique_ptr<NvBufSurfaceWrapper> nvbuf_surf, argus_cam.ConsumeFrame());

#ifdef NV_EGL_RENDERER
    int buf_fd = nvbuf_surf->fd();
    CHECK(buf_fd > 0) << "FD not valid.";
    renderer->render(buf_fd);
#endif

    GML_EXIT_IF_ERROR(nvbuf_surf->MapForCpu());

    const NvBufSurfaceParams& surf_params = nvbuf_surf->surface();
    nvbuf_surf->DumpInfo();

    const auto& h = surf_params.height;
    const auto& w = surf_params.width;
    CHECK_EQ(surf_params.planeParams.num_planes, 3);

    // TODO(oazizi): This is actually wrong, because the mapped addresses have a pitch
    //               that is different than the width. cv::Mat thus has the wrong data.
    cv::Mat y_plane(h, w, CV_8UC1, surf_params.mappedAddr.addr[0]);
    cv::Mat u_plane(h / 2, w / 2, CV_8UC1, surf_params.mappedAddr.addr[1]);
    cv::Mat v_plane(h / 2, w / 2, CV_8UC1, surf_params.mappedAddr.addr[2]);

    cv::Mat u_resized, v_resized;
    cv::Size size(w, h);
    cv::resize(u_plane, u_resized, size, 0, 0, cv::INTER_NEAREST);
    cv::resize(v_plane, v_resized, size, 0, 0, cv::INTER_NEAREST);

    VLOG(1) << absl::Substitute("$0x$1 $2x$3 $4x$5", y_plane.rows, y_plane.cols, u_resized.rows,
                                u_resized.cols, v_resized.rows, v_resized.cols);

    cv::Mat yuv;
    cv::merge(std::vector<cv::Mat>{y_plane, u_resized, v_resized}, yuv);

    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR);

    cv::imwrite(absl::StrFormat("/tmp/img%03d.jpg", i), yuv);

    // If you prefer to show this on screen, uncomment the following lines.
    // NOTE: our version of build of OpenCV doesn't currently support cv::imshow().
    // cv::imshow("test", bgr);
    // cv::pollKey();
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
