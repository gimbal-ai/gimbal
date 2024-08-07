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

#include "src/gem/calculators/core/text_stream_source_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {

constexpr std::string_view kPromptTag = "PROMPT";

absl::Status TextStreamSourceCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(core::ControlExecutionContextCalculator::UpdateContract(cc));

  cc->Outputs().Tag(kPromptTag).Set<std::string>();

  return absl::OkStatus();
}

Status TextStreamSourceCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                               exec::core::ControlExecutionContext* control_ctx) {
  auto ctrl_message = control_ctx->WaitForControlMessage(absl::Seconds(5));
  if (!ctrl_message) {
    return Status::OK();
  }
  auto prompt = ctrl_message->text_stream_control().prompt();

  // Use now as the timestamp.
  auto now = std::chrono::steady_clock::now();
  auto time_since_epoch = now.time_since_epoch();
  auto start_time_unix_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count();

  cc->Outputs()
      .Tag(kPromptTag)
      .Add(new std::string(prompt), mediapipe::Timestamp(start_time_unix_ms));

  return Status::OK();
}

REGISTER_CALCULATOR(TextStreamSourceCalculator);

}  // namespace gml::gem::calculators::core
