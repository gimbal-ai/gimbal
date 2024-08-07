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

#include "src/gem/calculators/core/text_stream_sink_calculator.h"

#include <memory>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::TextBatch;

constexpr std::string_view kTextTag = "TEXT_BATCH";
constexpr std::string_view kEOSTag = "EOS";

absl::Status TextStreamSinkCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(core::ControlExecutionContextCalculator::UpdateContract(cc));

  cc->Inputs().Tag(kTextTag).Set<std::string>();
  cc->Inputs().Tag(kEOSTag).Set<bool>();

  return absl::OkStatus();
}

Status TextStreamSinkCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                             exec::core::ControlExecutionContext* control_ctx) {
  std::vector<std::unique_ptr<google::protobuf::Message>> messages;

  auto batch = std::make_unique<TextBatch>();
  if (cc->Inputs().HasTag(kTextTag)) {
    batch->set_text(cc->Inputs().Tag(kTextTag).Get<std::string>());
  }
  if (cc->Inputs().HasTag(kEOSTag)) {
    batch->set_eos(cc->Inputs().Tag(kEOSTag).Get<bool>());
  }
  messages.push_back(std::move(batch));

  auto cb = control_ctx->GetMediaStreamCallback();
  if (!!cb) {
    GML_RETURN_IF_ERROR(cb(messages));
  }
  return Status::OK();
}

REGISTER_CALCULATOR(TextStreamSinkCalculator);

}  // namespace gml::gem::calculators::core
