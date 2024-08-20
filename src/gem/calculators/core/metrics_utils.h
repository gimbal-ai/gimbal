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

#include <mediapipe/framework/calculator_framework.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

namespace gml::gem::calculators::core::metrics_utils {

// We use an approximately exponential set of bounds for latency histogram, where the smallest bound
// is 100ns and the largest bound is 1000s.
const static std::vector<double> kLatencySecondsBucketBounds = {
    0,      1e-7, 2.5e-7, 5e-7,   1e-6,  2.5e-6, 5e-6,  1e-5,  2.5e-5, 5e-5,   1e-4,
    2.5e-4, 5e-4, 0.001,  0.0025, 0.005, 0.01,   0.025, 0.05,  0.1,    0.25,   0.5,
    1.0,    2.5,  5.0,    10.0,   25.0,  50.0,   100.0, 250.0, 500.0,  1000.0,
};

}  // namespace gml::gem::calculators::core::metrics_utils
