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

#include "src/gem/exec/plugin/huggingface/tokenizer_wrapper/hf_fast_tokenizer.h"

#include <memory>
#include <string>

#include "src/common/base/macros.h"
#include "src/gem/exec/core/tokenizer.h"

namespace gml::gem::exec {

using ::gml::gem::exec::core::Tokenizer;

std::unique_ptr<Tokenizer> HFFastTokenizer::Create(const std::string& json_path) {
  return std::unique_ptr<Tokenizer>(new HFFastTokenizer(json_path));
}

HFFastTokenizer::HFFastTokenizer(const std::string& json_path)
    : tokenizer_(new_tokenizer(json_path)) {}

std::vector<int> HFFastTokenizer::Encode(const std::string& text) const {
  rust::Vec<int> result = tokenizer_->encode(text, true);
  return {result.begin(), result.end()};
}

std::string HFFastTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string s(tokenizer_->decode(tokens, true));
  return s;
}

}  // namespace gml::gem::exec
