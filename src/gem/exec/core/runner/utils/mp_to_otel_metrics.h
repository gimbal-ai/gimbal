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
#include <opentelemetry/exporters/otlp/otlp_metric_utils.h>

#include "src/common/base/base.h"

namespace gml::gem::utils {

/**
 * Converts MediaPipe's profile stats into OTel metrics format.
 *
 * Function will try to forge through any recoverable errors, but will still return an error
 * status for the *first* error that was encountered. The resulting proto *may* still be usable
 * even if the translation error was encountered.
 */
Status CalculatorProfileVecToOTelProto(
    const std::vector<mediapipe::CalculatorProfile>& profiles, int64_t start_time_unix_nanos,
    opentelemetry::proto::metrics::v1::ResourceMetrics* metrics_out);

}  // namespace gml::gem::utils
