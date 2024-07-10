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

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <NvBufSurface.h>
#include <absl/strings/str_format.h>
#include <opencv4/opencv2/opencv.hpp>

#include "src/gem/devices/camera/argus/argus_cam.h"
#include "src/gem/devices/camera/argus/argus_manager.h"

constexpr int kNumFrames = 100;
const std::string kDeviceUUID = "";

int main(int argc, char** argv) {
  gml::EnvironmentGuard env_guard(&argc, argv);

  using gml::gem::devices::argus::ArgusCam;
  using gml::gem::devices::argus::ArgusManager;
  using gml::gem::devices::argus::NvBufSurfaceWrapper;

  auto& manager = ArgusManager::GetInstance();

  auto argus_cam_or_s = manager.GetCamera(kDeviceUUID);
  if (!argus_cam_or_s.ok()) {
    LOG(ERROR) << argus_cam_or_s.msg();
    return 1;
  }
  auto argus_cam = argus_cam_or_s.ConsumeValueOrDie();
  if (!argus_cam->Init().ok()) {
    argus_cam->Stop();
    return 1;
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < kNumFrames; ++i) {
    LOG(INFO) << absl::Substitute("--- $0", i);
    GML_ASSIGN_OR(std::unique_ptr<NvBufSurfaceWrapper> nvbuf_surf, argus_cam->ConsumeFrame(), {
      argus_cam->Stop();
      return 1;
    });

    if (!nvbuf_surf->MapForCpu().ok()) {
      argus_cam->Stop();
      return 1;
    }

    const NvBufSurfaceParams& surf_params = nvbuf_surf->surface();
    nvbuf_surf->DumpInfo();

    const auto& h = surf_params.height;
    const auto& w = surf_params.width;

    CHECK_EQ(surf_params.planeParams.num_planes, 3);
    CHECK_EQ(surf_params.planeParams.height[0], h);
    CHECK_EQ(surf_params.planeParams.width[0], w);
    CHECK_EQ(surf_params.planeParams.height[1], h / 2);
    CHECK_EQ(surf_params.planeParams.width[1], w / 2);
    CHECK_EQ(surf_params.planeParams.height[2], h / 2);
    CHECK_EQ(surf_params.planeParams.width[2], w / 2);

    cv::Mat y_plane(surf_params.planeParams.height[0], surf_params.planeParams.width[0], CV_8UC1,
                    surf_params.mappedAddr.addr[0], surf_params.planeParams.pitch[0]);
    cv::Mat u_plane(surf_params.planeParams.height[1], surf_params.planeParams.width[1], CV_8UC1,
                    surf_params.mappedAddr.addr[1], surf_params.planeParams.pitch[1]);
    cv::Mat v_plane(surf_params.planeParams.height[2], surf_params.planeParams.width[2], CV_8UC1,
                    surf_params.mappedAddr.addr[1], surf_params.planeParams.pitch[2]);

    cv::Mat uv_plane;
    cv::Mat yuv_plane;
    cv::hconcat(u_plane, v_plane, uv_plane);
    cv::vconcat(y_plane, uv_plane, yuv_plane);

    cv::Mat bgr;
    cv::cvtColor(yuv_plane, bgr, cv::COLOR_YUV2BGR_IYUV);

    cv::imwrite(absl::StrFormat("/tmp/img%03d.jpg", i), bgr);

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

  argus_cam->Stop();

  return 0;
}
