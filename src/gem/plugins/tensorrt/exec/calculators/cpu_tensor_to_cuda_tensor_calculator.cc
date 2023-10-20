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
#include "src/gem/plugins/tensorrt/exec/calculators/cpu_tensor_to_cuda_tensor_calculator.h"
#include "src/gem/plugins/tensorrt/exec/context.h"
#include "src/gem/plugins/tensorrt/exec/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace tensorrt {
namespace calculators {

absl::Status CPUTensorToCUDATensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<cputensor::CPUTensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<CUDATensorPtr>();
  }
  return absl::OkStatus();
}

Status CPUTensorToCUDATensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status CPUTensorToCUDATensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                    ExecutionContext* exec_ctx) {
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    auto cpu_tensor = cc->Inputs().Index(i).Get<cputensor::CPUTensorPtr>();

    GML_ASSIGN_OR_RETURN(auto cuda_tensor, exec_ctx->TensorPool()->GetTensor(cpu_tensor->size()));

    if (cudaMemcpy(static_cast<uint8_t*>(cuda_tensor->data()), cpu_tensor->data(),
                   cpu_tensor->size(), cudaMemcpyHostToDevice) != cudaSuccess) {
      return Status(types::CODE_INTERNAL, "Failed to memcpy from cuda host to device");
    }

    auto packet = mediapipe::MakePacket<CUDATensorPtr>(std::move(cuda_tensor));
    packet = packet.At(cc->InputTimestamp());
    cc->Outputs().Index(i).AddPacket(std::move(packet));
  }
  return Status::OK();
}

Status CPUTensorToCUDATensorCalculator::CloseImpl(mediapipe::CalculatorContext*,
                                                  ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(CPUTensorToCUDATensorCalculator);

}  // namespace calculators
}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
