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

namespace {
ov::Shape OVShapeFromCPUTensor(const exec::core::TensorShape& shape) {
  ov::Shape ov_shape(shape.begin(), shape.end());
  return ov_shape;
}

exec::core::DataType DataTypeFromOVType(const ov::element::Type& type) {
  switch (type) {
    case ov::element::Type_t::i32:
      return exec::core::DataType::INT32;
    case ov::element::Type_t::i64:
      return exec::core::DataType::INT64;
    case ov::element::Type_t::i8:
      return exec::core::DataType::INT8;
    case ov::element::Type_t::f32:
      return exec::core::DataType::FLOAT32;
    default:
      LOG(ERROR) << "Unsupported OpenVINO data type: " << type.get_type_name();
      return exec::core::DataType::UNKNOWN;
  }
}
}  // namespace

absl::Status OpenVinoExecuteCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  cc->InputSidePackets().Tag(kOpenVinoContextTag).Set<exec::core::ExecutionContext*>();
  cc->InputSidePackets().Tag(kCPUContextTag).Set<exec::core::ExecutionContext*>();
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<CPUTensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<CPUTensorPtr>();
  }
  cc->SetTimestampOffset(0);
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
    input_tensors.emplace_back(input_port.get_element_type(),
                               OVShapeFromCPUTensor(cpu_tensor->Shape()), cpu_tensor->data());
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
    cpu_tensor->SetDataType(DataTypeFromOVType(output_tensor.get_element_type()));

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
