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

#include <mediapipe/calculators/core/concatenate_vector_calculator.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/api/corepb/v1/mediastream.pb.h"

using ::gml::internal::api::core::v1::Detection;

namespace gml::gem::calculators::core {

using ConcatenateDetectionsCalculator = mediapipe::ConcatenateVectorCalculator<Detection>;

REGISTER_CALCULATOR(ConcatenateDetectionsCalculator);

}  // namespace gml::gem::calculators::core
