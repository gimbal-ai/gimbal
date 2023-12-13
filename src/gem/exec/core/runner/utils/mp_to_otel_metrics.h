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

#include <mediapipe/framework/calculator_framework.h>
#include <opentelemetry/exporters/otlp/otlp_metric_utils.h>

#include "src/common/base/base.h"

namespace gml::gem::utils {

/**
 * Converts MediaPipe's profile stats into OTel metrics format.
 *
 * Function will try to forge through any recoverable errors, but will still return an error
 *  status for the *first* error that was encountered. The resulting proto *may* still be usable
 * even if the translation error was encountered.
 */
Status CalculatorProfileVecToOTelProto(
    const std::vector<mediapipe::CalculatorProfile>& profiles,
    opentelemetry::proto::metrics::v1::ResourceMetrics* metrics_out);

}  // namespace gml::gem::utils
