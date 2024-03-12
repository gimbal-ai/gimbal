/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
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
