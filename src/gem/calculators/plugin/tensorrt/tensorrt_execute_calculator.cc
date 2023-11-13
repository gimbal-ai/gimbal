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

#include <absl/strings/substitute.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/common/base/base.h"
#include "src/gem/calculators/plugin/tensorrt/tensorrt_execute_calculator.h"
#include "src/gem/exec/plugin/tensorrt/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml::gem::calculators::tensorrt {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::tensorrt::CUDATensorPtr;
using ::gml::gem::exec::tensorrt::ExecutionContext;

namespace internal {
void CUDATensorPoolOutputAllocator::notifyShape(const char* tensor_name,
                                                const nvinfer1::Dims& dims) noexcept {
  auto tensor = outputs_[tensor_name];
  if (tensor == nullptr) {
    LOG(ERROR) << absl::Substitute(
        "TensorRT notified shape for tensor '$0' that is not in output allocator.", tensor_name);
    return;
  }

  TensorShape shape;
  for (int i = 0; i < dims.nbDims; ++i) {
    shape.push_back(dims.d[i]);
  }
  auto s = tensor->Reshape(shape);
  if (!s.ok()) {
    LOG(ERROR) << absl::Substitute("Failed to reshape CUDATensor '$0'", tensor_name);
    return;
  }
}

void* CUDATensorPoolOutputAllocator::reallocateOutput(const char* tensor_name, void* current_memory,
                                                      uint64_t size, uint64_t alignment) noexcept {
  // We ignore current_memory because we never call setTensorAddress for output tensors. So if
  // `current_memory` is non-null it will point to a previous iteration's output tensor, which is
  // most likely already freed.
  GML_UNUSED(current_memory);

  // We ignore alignment for now. Since we allocate memory with cudaMalloc, I believe we shouldn't
  // run into alignment issues, but we should probably check.
  GML_UNUSED(alignment);

  // TODO(james): should figure out error checking here instead of using ConsumeValueOrDie.
  auto tensor = pool_->GetTensor(size).ConsumeValueOrDie();
  tensor->SetDataType(output_data_types_[tensor_name]);
  outputs_.emplace(tensor_name, tensor);
  return tensor->data();
}

StatusOr<CUDATensorPtr> CUDATensorPoolOutputAllocator::AcquireOutput(const std::string& name) {
  auto node_handle = outputs_.extract(name);
  if (node_handle.empty()) {
    return Status(
        types::CODE_INVALID_ARGUMENT,
        absl::Substitute("TensorRT did not allocate expected output tensor with name $0", name));
  }
  return node_handle.mapped();
}

void CUDATensorPoolOutputAllocator::SetDataType(const std::string& name, DataType data_type) {
  output_data_types_.emplace(name, data_type);
}

}  // namespace internal

namespace {
DataType TensorRTDataTypeToGML(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::FLOAT32;
    case nvinfer1::DataType::kINT32:
      return DataType::INT32;
    default:
      return DataType::UNKNOWN;
  }
}

}  // namespace

absl::Status TensorRTExecuteCalculator::GetContract(mediapipe::CalculatorContract* cc) {
  GML_ABSL_RETURN_IF_ERROR(ExecutionContextBaseCalculator::UpdateContract(cc));

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<CUDATensorPtr>();
  }
  for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
    cc->Outputs().Index(i).Set<CUDATensorPtr>();
  }
  return absl::OkStatus();
}

Status TensorRTExecuteCalculator::OpenImpl(mediapipe::CalculatorContext* cc,
                                           ExecutionContext* exec_ctx) {
  options_ = cc->Options<optionspb::TensorRTExecuteCalculatorOptions>();

  output_allocator_ =
      std::make_unique<internal::CUDATensorPoolOutputAllocator>(exec_ctx->TensorPool());

  for (int i = 0; i < exec_ctx->CUDAEngine()->getNbIOTensors(); ++i) {
    auto name = exec_ctx->CUDAEngine()->getIOTensorName(i);
    if (exec_ctx->CUDAEngine()->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      auto trt_data_type = exec_ctx->CUDAEngine()->getTensorDataType(name);
      output_allocator_->SetDataType(name, TensorRTDataTypeToGML(trt_data_type));
      exec_ctx->NVExecutionContext()->setOutputAllocator(name, output_allocator_.get());
    }
  }
  return Status::OK();
}

Status TensorRTExecuteCalculator::ProcessImpl(mediapipe::CalculatorContext* cc,
                                              ExecutionContext* exec_ctx) {
  for (const auto& [i, name] : Enumerate(options_.input_onnx_name())) {
    const auto& packet = cc->Inputs().Index(i).Value();
    if (packet.IsEmpty()) {
      // We require all inputs to exist.
      return error::InvalidArgument(
          "TensorRTExecuteCalculator expects all inputs to be valid at each timestamp");
    }
    exec_ctx->NVExecutionContext()->setTensorAddress(name.c_str(),
                                                     packet.Get<CUDATensorPtr>()->data());
  }

  if (!exec_ctx->NVExecutionContext()->enqueueV3(exec_ctx->CUDAStream())) {
    return error::Internal("Failed to enqueue model kernels to CUDA stream");
  }

  // TODO(james): we should be able to avoid stream synchronization here if we have more processing
  // to do on the output.
  auto err = cudaStreamSynchronize(exec_ctx->CUDAStream());
  if (err != cudaSuccess) {
    return error::Internal("Failed to synchronize cuda stream");
  }

  for (const auto& [i, name] : Enumerate(options_.output_onnx_name())) {
    GML_ASSIGN_OR_RETURN(auto tensor, output_allocator_->AcquireOutput(name));
    auto packet = mediapipe::MakePacket<CUDATensorPtr>(tensor).At(cc->InputTimestamp());
    cc->Outputs().Index(i).AddPacket(std::move(packet));
  }
  return Status::OK();
}

Status TensorRTExecuteCalculator::CloseImpl(mediapipe::CalculatorContext*, ExecutionContext*) {
  return Status::OK();
}

REGISTER_CALCULATOR(TensorRTExecuteCalculator);

}  // namespace gml::gem::calculators::tensorrt
