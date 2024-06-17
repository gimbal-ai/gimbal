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

namespace gml::gem::calculators::core {

class LabelMapper {
 public:
  LabelMapper() = default;
  int get_id(const std::string& label);

 private:
  absl::flat_hash_map<std::string, int> label_to_id_;
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
};

}  // namespace gml::gem::calculators::core
