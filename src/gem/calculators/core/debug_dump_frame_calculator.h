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

#include <chrono>

#include <mediapipe/framework/calculator_framework.h>

namespace gml::gem::calculators::core {

/**
 * DebugDumpFrameCalculator Graph API:
 *
 *  Inputs:
 *    IMAGE_FRAME  mediapipe::ImageFrame
 *  No outputs. It dumps the given image frame to disk.
 *
 */
class DebugDumpFrameCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  std::chrono::steady_clock::time_point last_dump_;
  bool first_;
};

}  // namespace gml::gem::calculators::core
