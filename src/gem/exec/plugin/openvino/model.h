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

#include <openvino/openvino.hpp>
#include "src/gem/exec/core/model.h"

namespace gml::gem::exec::openvino {
class Model : public core::Model {
 public:
  explicit Model(ov::CompiledModel&& compiled_model) : compiled_model_(compiled_model) {}

  ov::CompiledModel& OpenVinoModel() { return compiled_model_; }

 private:
  ov::CompiledModel compiled_model_;
};
}  // namespace gml::gem::exec::openvino
