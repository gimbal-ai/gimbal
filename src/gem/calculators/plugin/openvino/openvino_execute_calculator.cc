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

#include "src/gem/calculators/plugin/openvino/openvino_execute_calculator.h"

#include <mediapipe/framework/calculator_registry.h>
#include <openvino/openvino.hpp>

#include "src/gem/calculators/plugin/openvino/optionspb/openvino_execute_calculator_options.pb.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/tensor.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/exec/plugin/cpu_tensor/cpu_tensor.h"

namespace gml::gem::calculators::openvino {
using exec::cpu_tensor::CPUTensorPtr;
using exec::openvino::ExecutionContext;

constexpr std::string_view kOpenVinoContextTag = "OV_EXEC_CTX";
constexpr std::string_view kCPUContextTag = "CPU_EXEC_CTX";

absl::Status OpenVinoExecuteCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->InputSidePackets().Tag(kOpenVinoContextTag).Set<exec::core::ExecutionContext*>();
  cc->InputSidePackets().Tag(kCPUContextTag).Set<exec::core::ExecutionContext*>();
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<CPUTensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<CPUTensorPtr>();
  }
  return absl::OkStatus();
}

absl::Status OpenVinoExecuteCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<optionspb::OpenVinoExecuteCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status OpenVinoExecuteCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto* ov_exec_ctx = static_cast<ExecutionContext*>(
      cc->InputSidePackets().Tag(kOpenVinoContextTag).Get<exec::core::ExecutionContext*>());
  auto* cpu_exec_ctx = static_cast<exec::cpu_tensor::ExecutionContext*>(
      cc->InputSidePackets().Tag(kCPUContextTag).Get<exec::core::ExecutionContext*>());
  auto& model = ov_exec_ctx->OpenVinoModel();

  std::vector<ov::Tensor> input_tensors;
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    const auto& packet = cc->Inputs().Index(i).Value();
    auto cpu_tensor = packet.Get<CPUTensorPtr>();
    auto input_port = model.input(i);
    input_tensors.emplace_back(input_port.get_element_type(), input_port.get_shape(),
                               cpu_tensor->data());
  }
  auto req = model.create_infer_request();
  for (const auto& [i, tensor] : Enumerate(input_tensors)) {
    req.set_input_tensor(i, tensor);
  }

  req.start_async();
  req.wait();

  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    auto output_tensor = req.get_output_tensor(i);
    GML_ABSL_ASSIGN_OR_RETURN(auto cpu_tensor,
                              cpu_exec_ctx->TensorPool()->GetTensor(output_tensor.get_byte_size()));
    // TODO(james): we might be able to refactor CPUTensor such that we can avoid this copy.
    std::memcpy(cpu_tensor->data(), output_tensor.data(), output_tensor.get_byte_size());

    exec::core::TensorShape shape;
    for (const auto& dim : output_tensor.get_shape()) {
      shape.push_back(static_cast<int>(dim));
    }
    GML_ABSL_RETURN_IF_ERROR(cpu_tensor->Reshape(shape));

    auto packet = mediapipe::MakePacket<CPUTensorPtr>(cpu_tensor).At(cc->InputTimestamp());
    cc->Outputs().Index(i).AddPacket(std::move(packet));
  }
  return absl::OkStatus();
}

absl::Status OpenVinoExecuteCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenVinoExecuteCalculator);

}  // namespace gml::gem::calculators::openvino
