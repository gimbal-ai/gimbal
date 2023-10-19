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
#include <mediapipe/framework/calculator_base.h>

#include "src/common/base/base.h"
#include "src/gem/core/exec/context.h"

namespace gml {
namespace gem {
namespace core {
namespace calculators {

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
    cc->InputSidePackets().Tag(kExecutionContextTag).Set<ExecutionContext*>();
    return absl::OkStatus();
  }

  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    exec_ctx_ = static_cast<TExecutionContext*>(
        cc->InputSidePackets().Tag(kExecutionContextTag).Get<ExecutionContext*>());
    if (exec_ctx_ == nullptr) {
      return absl::Status(absl::StatusCode::kInvalidArgument, "EXEC_CTX is null");
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

}  // namespace calculators
}  // namespace core
}  // namespace gem
}  // namespace gml
