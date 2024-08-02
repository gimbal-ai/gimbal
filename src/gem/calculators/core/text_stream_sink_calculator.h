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

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::calculators::core {

/**
 *  TextStreamSinkCalculator Graph API:
 *
 *  Inputs:
 *    TEXT_BATCH std::string - Text to be written to the output stream.
 *    EOS bool - A flag indicating whether it is the end of the stream.
 *
 *  Outputs:
 *    This is a sink node so there are no data mediapipe outputs. Instead the node outputs proto
 *      data to the GEM controller through the ControlExecutionContext
 *
 */
class TextStreamSinkCalculator : public core::ControlExecutionContextCalculator {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     exec::core::ControlExecutionContext* control_ctx) override;
};

}  // namespace gml::gem::calculators::core
