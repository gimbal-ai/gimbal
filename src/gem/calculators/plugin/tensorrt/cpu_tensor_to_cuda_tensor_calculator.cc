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

#include "src/gem/calculators/plugin/tensorrt/cpu_tensor_to_cuda_tensor_calculator.h"

#include <cuda_runtime_api.h>

#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml::gem::calculators::tensorrt {

using ::gml::gem::exec::tensorrt::CUDATensorPtr;

absl::Status CPUTensorToCUDATensorCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<exec::cpu_tensor::CPUTensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<CUDATensorPtr>();
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

Status CPUTensorToCUDATensorCalculator::OpenImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

Status CPUTensorToCUDATensorCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                                    ExecutionContext* exec_ctx) {
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    auto cpu_tensor = cc->Inputs().Index(i).Get<exec::cpu_tensor::CPUTensorPtr>();

    GML_ASSIGN_OR_RETURN(auto cuda_tensor, exec_ctx->TensorPool()->GetTensor(cpu_tensor->size()));

    GML_RETURN_IF_ERROR(cuda_tensor->Reshape(cpu_tensor->Shape()));
    cuda_tensor->SetDataType(cpu_tensor->DataType());

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

}  // namespace gml::gem::calculators::tensorrt
