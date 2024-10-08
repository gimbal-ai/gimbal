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

#include "src/gem/calculators/plugin/argus/nvbuf_to_image_frame_calculator.h"

#include "libyuv/convert.h"
#include <absl/status/status.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>

#include "src/gem/devices/camera/argus/nvbufsurfwrapper.h"

namespace gml::gem::calculators::argus {

using devices::argus::NvBufSurfaceWrapper;

absl::Status NvBufSurfToImageFrameCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::shared_ptr<NvBufSurfaceWrapper>>();
  cc->Outputs().Index(0).Set<mediapipe::ImageFrame>();
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status NvBufSurfToImageFrameCalculator::Process(mediapipe::CalculatorContext* cc) {
  const auto& nvbuf_surf = cc->Inputs().Index(0).Get<std::shared_ptr<NvBufSurfaceWrapper>>();

  if (!nvbuf_surf->IsMapped()) {
    return absl::InternalError("The provided nvbuf is not mapped to CPU memory.");
  }

  const NvBufSurfaceParams& surf_params = nvbuf_surf->surface();
  if (surf_params.planeParams.num_planes != 3) {
    return absl::InternalError(
        absl::Substitute("Expecting 3 planes, but got $0.", surf_params.planeParams.num_planes));
  }

  auto image_frame = std::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::FORMAT_SRGB,
                                                             surf_params.width, surf_params.height);

  auto y_buf_ptr = reinterpret_cast<uint8_t*>(surf_params.mappedAddr.addr[0]);
  auto u_buf_ptr = reinterpret_cast<uint8_t*>(surf_params.mappedAddr.addr[1]);
  auto v_buf_ptr = reinterpret_cast<uint8_t*>(surf_params.mappedAddr.addr[2]);

  auto y_pitch = surf_params.planeParams.pitch[0];
  auto u_pitch = surf_params.planeParams.pitch[1];
  auto v_pitch = surf_params.planeParams.pitch[2];

  libyuv::I420ToRAW(                                              //
      y_buf_ptr, y_pitch,                                         //
      u_buf_ptr, u_pitch,                                         //
      v_buf_ptr, v_pitch,                                         //
      image_frame->MutablePixelData(), image_frame->WidthStep(),  //
      surf_params.width, surf_params.height);

  cc->Outputs().Index(0).Add(image_frame.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

REGISTER_CALCULATOR(NvBufSurfToImageFrameCalculator);

}  // namespace gml::gem::calculators::argus
