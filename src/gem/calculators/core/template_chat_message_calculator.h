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

#include <inja/inja.hpp>
#include <mediapipe/framework/calculator_framework.h>

#include "src/gem/calculators/core/optionspb/template_chat_message_calculator_options.pb.h"

namespace gml::gem::calculators::core {

/**
 * TemplateChatMessageCalculator Graph API:
 *
 *  Options:
 *    message_template The template to use for the chat message.
 *    preset_system_prompts Preset prompts that should be specified for the system at the beginning
 *    of the message.
 *    add_generation_prompt Whether to add a prompt to the end of the message.
 *  Inputs:
 *    QUERY string The user's query.
 *    DOCUMENTS <std::vector<string>> Additional content to include in the chat message on behalf of
 * the system. Outputs: TEXT string The templated chat message.
 *
 */
class TemplateChatMessageCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  std::string message_template_;
  std::string preset_system_prompt_;
  bool add_generation_prompt_;
};

}  // namespace gml::gem::calculators::core
