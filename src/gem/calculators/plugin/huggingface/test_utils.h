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

#include "src/gem/exec/core/tokenizer.h"

namespace gml::gem::calculators::huggingface {

using gml::gem::exec::core::Tokenizer;

class FakeTokenizer : public Tokenizer {
 public:
  FakeTokenizer(std::string& text, std::vector<int> token_ids)
      : text_(text), token_ids_(std::move(token_ids)){};

  std::vector<int> Encode(const std::string&) const override { return token_ids_; }

  std::string Decode(const std::vector<int>&) const override { return text_; }

 private:
  std::string text_;
  std::vector<int> token_ids_;
};

}  // namespace gml::gem::calculators::huggingface
