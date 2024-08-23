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

#include <memory>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::TextBatch;

/**
 *  TextStreamSinkCalculator Graph API:
 *
 *  Inputs:
 *    TEXT_BATCH std::string - Text to be written to the output stream.
 *    EOS bool - A flag indicating whether it is the end of the stream.
 *    PROMPT_TIMESTAMP mediapipe::Timestamp - The timestamp of the prompt the current token belongs
 * to.
 *
 *  Outputs:
 *    The node outputs proto data to the GEM controller through the ControlExecutionContext.
 *    A FINISHED bool packet is output when the EOS flag is set to true to indicate the response
 *    to a prompt is finished.
 */
class TextStreamSinkCalculator : public ControlExecutionContextCalculator {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    GML_ABSL_RETURN_IF_ERROR(ControlExecutionContextCalculator::UpdateContract(cc));

    cc->Inputs().Tag(kTextTag).Set<std::string>();
    cc->Inputs().Tag(kEOSTag).Set<bool>();

    if (cc->Inputs().HasTag(kPromptTimestampTag)) {
      cc->Inputs().Tag(kPromptTimestampTag).Set<mediapipe::Timestamp>();
    }

    if (cc->Outputs().HasTag(kFinishedTag)) {
      RET_CHECK(cc->Inputs().HasTag(kPromptTimestampTag))
          << absl::Substitute("$0 required when using $1.", kPromptTimestampTag, kFinishedTag);
      cc->Outputs().Tag(kFinishedTag).Set<bool>();
    }

    return absl::OkStatus();
  }

  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     exec::core::ControlExecutionContext* control_ctx) override {
    std::vector<std::unique_ptr<google::protobuf::Message>> messages;

    auto batch = std::make_unique<TextBatch>();
    batch->set_text(cc->Inputs().Tag(kTextTag).Get<std::string>());
    batch->set_eos(cc->Inputs().Tag(kEOSTag).Get<bool>());
    messages.push_back(std::move(batch));

    auto cb = control_ctx->GetMediaStreamCallback();
    if (!!cb) {
      GML_RETURN_IF_ERROR(cb(messages));
    }

    if (cc->Inputs().HasTag(kPromptTimestampTag)) {
      auto prompt_timestamp = cc->Inputs().Tag(kPromptTimestampTag).Get<mediapipe::Timestamp>();
      auto eos = cc->Inputs().Tag(kEOSTag).Get<bool>();
      if (eos) {
        cc->Outputs()
            .Tag(kFinishedTag)
            .AddPacket(mediapipe::MakePacket<bool>(true).At(prompt_timestamp));
      }
    }

    return Status::OK();
  }

 private:
  static constexpr std::string_view kTextTag = "TEXT_BATCH";
  static constexpr std::string_view kEOSTag = "EOS";
  static constexpr std::string_view kPromptTimestampTag = "PROMPT_TIMESTAMP";
  static constexpr std::string_view kFinishedTag = "FINISHED";
};

REGISTER_CALCULATOR(TextStreamSinkCalculator);

}  // namespace gml::gem::calculators::core
