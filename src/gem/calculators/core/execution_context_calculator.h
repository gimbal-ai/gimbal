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
#include <mediapipe/framework/calculator_base.h>

#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/control_context.h"

namespace gml::gem::calculators::core {

static constexpr std::string_view kExecutionContextTag = "EXEC_CTX";

/**
 * ExecutionContextCalculator is a convenience class for plugins to create calculators that require
 * plugin-specific ExecutionContexts.
 *
 * TExecutionContext must be a subclass of core::ExecutionContext.
 *
 * Subclasses of ExecutionContextCalculator<TExecutionContext> must call
 * ExecutionContextCalculator<TExecutionContext>::UpdateContract inside their GetContract static
 * method.
 *
 * The mediapipe node for any subclasses must have an input_side_packet tagged with EXEC_CTX.
 */
template <typename TExecutionContext>
class ExecutionContextCalculator : public mediapipe::CalculatorBase {
 protected:
  virtual Status OpenImpl(mediapipe::CalculatorContext*, TExecutionContext*) {
    return Status::OK();
  }
  virtual Status ProcessImpl(mediapipe::CalculatorContext*, TExecutionContext*) {
    return Status::OK();
  }
  virtual Status CloseImpl(mediapipe::CalculatorContext*, TExecutionContext*) {
    return Status::OK();
  }

 public:
  static absl::Status UpdateContract(mediapipe::CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kExecutionContextTag).Set<exec::core::ExecutionContext*>();
    return absl::OkStatus();
  }

  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    exec_ctx_ = static_cast<TExecutionContext*>(
        cc->InputSidePackets().Tag(kExecutionContextTag).Get<exec::core::ExecutionContext*>());
    if (exec_ctx_ == nullptr) {
      return {absl::StatusCode::kInvalidArgument, "EXEC_CTX is null"};
    }
    GML_ABSL_RETURN_IF_ERROR(OpenImpl(cc, exec_ctx_));
    return absl::OkStatus();
  }

  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(ProcessImpl(cc, exec_ctx_));
    return absl::OkStatus();
  }

  absl::Status Close(mediapipe::CalculatorContext* cc) override {
    GML_ABSL_RETURN_IF_ERROR(CloseImpl(cc, exec_ctx_));
    return absl::OkStatus();
  }

 private:
  TExecutionContext* exec_ctx_;
};

using ControlExecutionContextCalculator =
    ExecutionContextCalculator<exec::core::ControlExecutionContext>;

}  // namespace gml::gem::calculators::core
