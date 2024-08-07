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

#include "src/gem/calculators/core/template_chat_message_calculator.h"

#include <thread>
#include <variant>
#include <vector>

#include <absl/status/status.h>
#include <absl/strings/str_replace.h>
#include <inja/inja.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/timestamp.h>

namespace gml::gem::calculators::core {

using ::gml::gem::calculators::core::optionspb::TemplateChatMessageCalculatorOptions;

constexpr std::string_view kQueryTag = "QUERY";
constexpr std::string_view kDocumentsTag = "DOCUMENTS";
constexpr std::string_view kTextTag = "TEXT";

absl::Status TemplateChatMessageCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kQueryTag).Set<std::string>();
  cc->Inputs().Tag(kDocumentsTag).Set<std::vector<std::string>>();

  cc->Outputs().Tag(kTextTag).Set<std::string>();

  return absl::OkStatus();
}

absl::Status TemplateChatMessageCalculator::Open(mediapipe::CalculatorContext* cc) {
  auto& options = cc->Options<TemplateChatMessageCalculatorOptions>();

  // Format the jinja template for inja's expected template.
  // Any ' should be replaced with an escaped \", and any dictionary accesses should be replaced
  // with a dot.
  message_template_ = absl::StrReplaceAll(options.message_template(),
                                          {{"\n", "\\n"}, {"'", "\""}, {"['", "."}, {"']", ""}});

  preset_system_prompt_ = options.preset_system_prompt();
  add_generation_prompt_ = options.add_generation_prompt();

  return absl::OkStatus();
}

absl::Status TemplateChatMessageCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto query = cc->Inputs().Tag(kQueryTag).Get<std::string>();
  auto content = cc->Inputs().Tag(kDocumentsTag).Get<std::vector<std::string>>();
  inja::Environment env;
  inja::json data;
  data["add_generation_prompt"] = add_generation_prompt_;
  data["messages"] = inja::json::array();

  data["messages"].push_back(inja::json{{"role", "system"}, {"content", preset_system_prompt_}});
  for (const auto& prompt : content) {
    data["messages"].push_back(inja::json{{"role", "system"}, {"content", prompt}});
  }
  data["messages"].push_back(inja::json{{"role", "user"}, {"content", query}});

  try {
    std::string output =
        env.render(message_template_, data);  // NOLINT(clang-analyzer-core.StackAddressEscape)

    cc->Outputs().Tag(kTextTag).AddPacket(
        mediapipe::MakePacket<std::string>(output).At(cc->InputTimestamp()));
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }

  return absl::OkStatus();
}

REGISTER_CALCULATOR(TemplateChatMessageCalculator);

}  // namespace gml::gem::calculators::core
