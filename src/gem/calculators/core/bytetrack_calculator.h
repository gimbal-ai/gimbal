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

#include <ByteTrack/BYTETracker.h>

#include <mediapipe/framework/calculator_framework.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/bimap.h"
#include "src/common/base/statusor.h"
#include "src/gem/calculators/core/optionspb/bytetrack_calculator_options.pb.h"

namespace gml::gem::calculators::core {

class LabelMapper {
 public:
  LabelMapper() = default;
  int get_id(const std::string& label);
  StatusOr<std::string> get_label(int id);

 private:
  BiMap<std::string, int> label_id_bimap_;
};

class ByteTrackCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  std::unique_ptr<byte_track::BYTETracker> byte_tracker_ = nullptr;
  LabelMapper label_mapper_;
  optionspb::ByteTrackCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::core
