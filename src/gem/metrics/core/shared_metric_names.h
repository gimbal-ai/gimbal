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

#include <string_view>

namespace gml::gem::metrics::core {

/**
 * Any metric that can be produced by different plugins should keep the metric name here.
 */

constexpr std::string_view kGPUMemoryGEMUsageGaugeName = "gml.gem.gpu.memory.usage.bytes";
// Size of GPU memory per GPU device on the system.
constexpr std::string_view kGPUMemorySystemSizeGaugeName = "gml.system.gpu.memory.size.bytes";
constexpr std::string_view kGPUMemorySystemUsageGaugeName = "gml.system.gpu.memory.usage.bytes";

constexpr std::string_view kGPUUtilizationSystemCounterName = "gml.system.gpu.seconds.total";
constexpr std::string_view kGPUUtilizationGEMCounterName = "gml.gem.gpu.seconds.total";

}  // namespace gml::gem::metrics::core
