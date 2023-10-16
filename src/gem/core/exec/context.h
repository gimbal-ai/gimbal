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

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace core {

/**
 * ExecutionContext represents a plugin-specific context to be passed to plugin calculators.
 * From gem::core's perspective, its just an opaque pointer that will be added as an
 * input_side_packet to any mediapipe calculators that request it.
 */
class ExecutionContext {
 public:
  virtual ~ExecutionContext() {}
};

}  // namespace core
}  // namespace gem
}  // namespace gml
