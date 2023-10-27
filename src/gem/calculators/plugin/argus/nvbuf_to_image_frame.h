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

#include <mediapipe/framework/calculator_framework.h>

namespace gml {
namespace gem {
namespace calculators {
namespace argus {

/**
 * NvBufSurfToImageFrameCalculator API:
 *
 *  Input Side Packets:
 *    None
 *  Inputs:
 *    0 NvBufSurfaceWrapper The input frame in YUV420 3-plane format (I420).
 *  Outputs:
 *    0 mediapipe::ImageFrame The input frame converted to mediapipe's RGB format.
 *
 */
class NvBufSurfToImageFrameCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;
};

}  // namespace argus
}  // namespace calculators
}  // namespace gem
}  // namespace gml