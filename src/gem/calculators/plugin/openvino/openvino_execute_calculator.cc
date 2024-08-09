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

#include "src/common/base/error.h"
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
constexpr std::string_view kResetStateTag = "RESET_STATE";
constexpr std::string_view kInputOutputTag = "";

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
  for (int i = 0; i < cc->Inputs().NumEntries(kInputOutputTag); ++i) {
    cc->Inputs().Index(i).Set<CPUTensorPtr>();
  }

  if (cc->Inputs().HasTag(kResetStateTag)) {
    cc->Inputs().Tag(kResetStateTag).Set<bool>();
  }

  for (int i = 0; i < cc->Outputs().NumEntries(kInputOutputTag); ++i) {
    cc->Outputs().Index(i).Set<CPUTensorPtr>();
  }
  cc->SetTimestampOffset(0);
  return absl::OkStatus();
}

absl::Status OpenVinoExecuteCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<optionspb::OpenVinoExecuteCalculatorOptions>();
  auto* ov_exec_ctx = static_cast<ExecutionContext*>(
      cc->InputSidePackets().Tag(kOpenVinoContextTag).Get<exec::core::ExecutionContext*>());
  auto& model = ov_exec_ctx->OpenVinoModel();
  infer_request_ = model.create_infer_request();

  GML_ABSL_RETURN_IF_ERROR(SetupIndices(cc));

  return absl::OkStatus();
}

Status OpenVinoExecuteCalculator::SetupIndices(mediapipe::CalculatorContext* cc) {
  auto* ov_exec_ctx = static_cast<ExecutionContext*>(
      cc->InputSidePackets().Tag(kOpenVinoContextTag).Get<exec::core::ExecutionContext*>());
  auto& model = ov_exec_ctx->OpenVinoModel();

  if (options_.loopback_input_indices_size() != options_.loopback_output_indices_size()) {
    return gml::error::InvalidArgument(
        "loopback_input_indices and loopback_output_indices must have the same size ($0 vs $1)",
        options_.loopback_input_indices_size(), options_.loopback_output_indices_size());
  }

  auto num_model_inputs = static_cast<int64_t>(model.inputs().size());
  absl::flat_hash_set<int64_t> loopback_inputs(options_.loopback_input_indices().begin(),
                                               options_.loopback_input_indices().end());
  for (int64_t i = 0; i < num_model_inputs; ++i) {
    if (loopback_inputs.contains(i)) {
      continue;
    }
    calc_input_idx_to_model_idx_.push_back(i);
  }
  auto num_model_outputs = static_cast<int64_t>(model.outputs().size());
  absl::flat_hash_set<int64_t> loopback_outputs(options_.loopback_output_indices().begin(),
                                                options_.loopback_output_indices().end());
  for (int64_t i = 0; i < num_model_outputs; ++i) {
    if (loopback_outputs.contains(i)) {
      continue;
    }
    calc_output_idx_to_model_idx_.push_back(i);
  }

  for (int i = 0; i < options_.loopback_input_indices_size(); ++i) {
    loopback_mapping_.emplace(options_.loopback_input_indices(i),
                              options_.loopback_output_indices(i));
  }

  if (calc_input_idx_to_model_idx_.size() !=
      static_cast<size_t>(cc->Inputs().NumEntries(kInputOutputTag))) {
    return gml::error::InvalidArgument(
        "OpenVINOExecuteCalculator has $0 inputs but the model expects $1 (non-loopback) inputs",
        cc->Inputs().NumEntries(kInputOutputTag), calc_input_idx_to_model_idx_.size());
  }
  if (calc_output_idx_to_model_idx_.size() !=
      static_cast<size_t>(cc->Outputs().NumEntries(kInputOutputTag))) {
    return gml::error::InvalidArgument(
        "OpenVINOExecuteCalculator has $0 outputs but the model expects $1 (non-loopback) outputs",
        cc->Outputs().NumEntries(kInputOutputTag), calc_output_idx_to_model_idx_.size());
  }

  if ((calc_input_idx_to_model_idx_.size() + loopback_mapping_.size()) !=
      static_cast<size_t>(num_model_inputs)) {
    return gml::error::InvalidArgument(
        "OpenVINOExecuteCalculator has $0 inputs and $1 loopback inputs but model has $2 inputs",
        calc_input_idx_to_model_idx_.size(), loopback_mapping_.size(), num_model_inputs);
  }
  if ((calc_output_idx_to_model_idx_.size() + loopback_mapping_.size()) !=
      static_cast<size_t>(num_model_outputs)) {
    return gml::error::InvalidArgument(
        "OpenVINOExecuteCalculator has $0 outputs and $1 loopback outputs but model has $2 outputs",
        calc_input_idx_to_model_idx_.size(), loopback_mapping_.size(), num_model_inputs);
  }
  return Status::OK();
}

absl::Status OpenVinoExecuteCalculator::Process(mediapipe::CalculatorContext* cc) {
  auto* ov_exec_ctx = static_cast<ExecutionContext*>(
      cc->InputSidePackets().Tag(kOpenVinoContextTag).Get<exec::core::ExecutionContext*>());
  auto* cpu_exec_ctx = static_cast<exec::cpu_tensor::ExecutionContext*>(
      cc->InputSidePackets().Tag(kCPUContextTag).Get<exec::core::ExecutionContext*>());
  auto& model = ov_exec_ctx->OpenVinoModel();

  if (cc->Inputs().HasTag(kResetStateTag) && cc->Inputs().Tag(kResetStateTag).Get<bool>()) {
    infer_request_ = model.create_infer_request();
  }

  std::vector<ov::Tensor> input_tensors;
  for (int i = 0; i < cc->Inputs().NumEntries(kInputOutputTag); ++i) {
    const auto& packet = cc->Inputs().Index(i).Value();
    auto cpu_tensor = packet.Get<CPUTensorPtr>();
    auto model_input_idx = calc_input_idx_to_model_idx_[i];
    auto input_port = model.input(model_input_idx);
    auto& ov_tensor =
        input_tensors.emplace_back(input_port.get_element_type(),
                                   OVShapeFromCPUTensor(cpu_tensor->Shape()), cpu_tensor->data());
    infer_request_.set_input_tensor(model_input_idx, ov_tensor);
  }

  infer_request_.start_async();
  infer_request_.wait();

  for (int i = 0; i < cc->Outputs().NumEntries(kInputOutputTag); ++i) {
    auto model_output_idx = calc_output_idx_to_model_idx_[i];
    auto output_tensor = infer_request_.get_output_tensor(model_output_idx);
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

  for (const auto& [loopback_in_idx, loopback_out_idx] : loopback_mapping_) {
    infer_request_.set_input_tensor(loopback_in_idx,
                                    infer_request_.get_output_tensor(loopback_out_idx));
  }

  return absl::OkStatus();
}

absl::Status OpenVinoExecuteCalculator::Close(mediapipe::CalculatorContext*) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenVinoExecuteCalculator);

}  // namespace gml::gem::calculators::openvino
