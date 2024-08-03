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

#include <memory>
#include <string>
#include <vector>

#include "src/gem/exec/core/tokenizer.h"
#include "src/gem/exec/plugin/huggingface/tokenizer_wrapper/src/lib.rs.h"

namespace gml::gem::exec {

// HFFastTokenizer is a wrapper around the Rust implementation of the Hugging Face tokenizer.
// It implements the Tokenizer interface.
class HFFastTokenizer : public gml::gem::exec::core::Tokenizer {
 public:
  HFFastTokenizer() = delete;
  ~HFFastTokenizer() override = default;
  static std::unique_ptr<gml::gem::exec::core::Tokenizer> Create(const std::string& json_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  explicit HFFastTokenizer(const std::string& json_path);

 private:
  rust::Box<TokenizerWrapper> tokenizer_;
};

}  // namespace gml::gem::exec
