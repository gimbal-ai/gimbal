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

#include <cuda_runtime_api.h>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/plugins/tensorrt/exec/calculators/cuda_tensor_to_cpu_tensor_calculator.h"
#include "src/gem/plugins/tensorrt/exec/context.h"
#include "src/gem/plugins/tensorrt/exec/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace tensorrt {
namespace calculators {

absl::Status CUDATensorToCPUTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(
      core::calculators::ExecutionContextCalculator<cputensor::ExecutionContext>::UpdateContract(
          cc));

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<CUDATensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<cputensor::CPUTensorPtr>();
  }
  return absl::OkStatus();
}

Status CUDATensorToCPUTensorCalculator::OpenImpl(mediapipe::CalculatorContext*,
                                                 cputensor::ExecutionContext*) {
  return Status::OK();
}

Status CUDATensorToCPUTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                    cputensor::ExecutionContext* exec_ctx) {
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    auto cuda_tensor = cc->Inputs().Index(i).Get<CUDATensorPtr>();

    GML_ASSIGN_OR_RETURN(auto cpu_tensor, exec_ctx->TensorPool()->GetTensor(cuda_tensor->size()));

    if (cudaMemcpy(cpu_tensor->data(), static_cast<uint8_t*>(cuda_tensor->data()),
                   cuda_tensor->size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return Status(types::CODE_INTERNAL, "Failed to memcpy from cuda device to host");
    }

    auto packet = mediapipe::MakePacket<cputensor::CPUTensorPtr>(std::move(cpu_tensor));
    packet = packet.At(cc->InputTimestamp());
    cc->Outputs().Index(i).AddPacket(std::move(packet));
  }
  return Status::OK();
}

Status CUDATensorToCPUTensorCalculator::CloseImpl(mediapipe::CalculatorContext*,
                                                  cputensor::ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(CUDATensorToCPUTensorCalculator);

}  // namespace calculators
}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
