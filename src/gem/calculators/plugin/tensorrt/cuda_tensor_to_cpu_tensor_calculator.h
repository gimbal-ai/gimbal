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

#include "src/gem/calculators/plugin/tensorrt/base.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml::gem::calculators::tensorrt {

/**
 * CUDATensorToCPUTensorCalculator Graph API:
 *
 *  Input Side Packets:
 *    EXEC_CTX cputensor::ExecutionContext*
 *  Inputs:
 *    Each input must be a CUDATensorPtr
 *  Outputs:
 *    Each output will be a CPUTensorPtr
 *
 */
class CUDATensorToCPUTensorCalculator
    : public core::ExecutionContextCalculator<exec::cpu_tensor::ExecutionContext> {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc,
                  exec::cpu_tensor::ExecutionContext* exec_ctx) override;
  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     exec::cpu_tensor::ExecutionContext* exec_ctx) override;
  Status CloseImpl(mediapipe::CalculatorContext* cc,
                   exec::cpu_tensor::ExecutionContext* exec_ctx) override;
};

}  // namespace gml::gem::calculators::tensorrt
