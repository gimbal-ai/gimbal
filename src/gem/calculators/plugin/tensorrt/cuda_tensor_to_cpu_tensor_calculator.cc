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
#include "src/gem/calculators/plugin/tensorrt/cuda_tensor_to_cpu_tensor_calculator.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace calculators {
namespace tensorrt {

using ::gml::gem::exec::tensorrt::CUDATensorPtr;

absl::Status CUDATensorToCPUTensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(
      core::ExecutionContextCalculator<exec::cpu_tensor::ExecutionContext>::UpdateContract(cc));

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<CUDATensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<exec::cpu_tensor::CPUTensorPtr>();
  }
  return absl::OkStatus();
}

Status CUDATensorToCPUTensorCalculator::OpenImpl(mediapipe::CalculatorContext*,
                                                 exec::cpu_tensor::ExecutionContext*) {
  return Status::OK();
}

Status CUDATensorToCPUTensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                    exec::cpu_tensor::ExecutionContext* exec_ctx) {
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    auto cuda_tensor = cc->Inputs().Index(i).Get<CUDATensorPtr>();

    GML_ASSIGN_OR_RETURN(auto cpu_tensor, exec_ctx->TensorPool()->GetTensor(cuda_tensor->size()));

    GML_RETURN_IF_ERROR(cpu_tensor->Reshape(cuda_tensor->Shape()));
    cpu_tensor->SetDataType(cuda_tensor->DataType());

    if (cudaMemcpy(cpu_tensor->data(), static_cast<uint8_t*>(cuda_tensor->data()),
                   cuda_tensor->size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return Status(types::CODE_INTERNAL, "Failed to memcpy from cuda device to host");
    }

    auto packet = mediapipe::MakePacket<exec::cpu_tensor::CPUTensorPtr>(std::move(cpu_tensor));
    packet = packet.At(cc->InputTimestamp());
    cc->Outputs().Index(i).AddPacket(std::move(packet));
  }
  return Status::OK();
}

Status CUDATensorToCPUTensorCalculator::CloseImpl(mediapipe::CalculatorContext*,
                                                  exec::cpu_tensor::ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(CUDATensorToCPUTensorCalculator);

}  // namespace tensorrt
}  // namespace calculators
}  // namespace gem
}  // namespace gml
