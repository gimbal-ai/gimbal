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

#include "src/common/base/base.h"

namespace gml::gem::exec::core {

/**
 * ExecutionContext represents a plugin-specific context to be passed to plugin calculators.
 * From gem::core's perspective, its just an opaque pointer that will be added as an
 * input_side_packet to any mediapipe calculators that request it.
 */
class ExecutionContext {
 public:
  virtual ~ExecutionContext() = default;
};

}  // namespace gml::gem::exec::core
