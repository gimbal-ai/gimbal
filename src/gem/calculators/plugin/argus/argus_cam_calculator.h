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

#include <mediapipe/framework/calculator_framework.h>

#include "src/common/metrics/metrics_system.h"
#include "src/gem/calculators/plugin/argus/optionspb/argus_cam_calculator_options.pb.h"
#include "src/gem/devices/camera/argus/argus_cam.h"
#include "src/gem/devices/camera/argus/argus_manager.h"

namespace gml::gem::calculators::argus {

class ArgusCamSourceCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  optionspb::ArgusCamSourceCalculatorOptions options_;
  devices::argus::ArgusManager::ArgusCamManagedPtr argus_cam_;

  // Metrics.
  opentelemetry::metrics::Gauge<double>* fps_gauge_;
};

}  // namespace gml::gem::calculators::argus
