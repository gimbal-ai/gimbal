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

#include "src/gem/exec/core/context.h"
#include "src/gem/exec/plugin/openvino/model.h"

namespace gml::gem::exec::openvino {

class ExecutionContext : public core::ExecutionContext {
 public:
  explicit ExecutionContext(core::Model* model) : model_(static_cast<Model*>(model)) {}

  ov::CompiledModel& OpenVinoModel() { return model_->OpenVinoModel(); }

 private:
  Model* model_;
};

}  // namespace gml::gem::exec::openvino
