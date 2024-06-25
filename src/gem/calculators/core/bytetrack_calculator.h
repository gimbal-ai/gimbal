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
