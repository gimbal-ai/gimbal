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

#include "src/common/base/base.h"
#include "src/gem/exec/core/model.h"

namespace gml::gem::exec::core {

/**
 * Tokenizer is the base class for a tokenizer model.
 */
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  static std::unique_ptr<Tokenizer> Create(const std::string& json_path);

  virtual std::vector<int> Encode(const std::string& text) const = 0;

  virtual std::string Decode(const std::vector<int>& tokens) const = 0;
};

}  // namespace gml::gem::exec::core
