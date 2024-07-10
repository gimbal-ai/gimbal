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
