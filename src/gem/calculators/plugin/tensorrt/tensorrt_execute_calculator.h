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

#include <NvInfer.h>

#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/calculators/plugin/tensorrt/base.h"
#include "src/gem/calculators/plugin/tensorrt/optionspb/tensorrt_execute_calculator_options.pb.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml::gem::calculators::tensorrt {

namespace internal {
/**
 * CUDATensorPoolOutputAllocator implements TensorRT's IOutputAllocator to dynamically allocate
 * CUDATensors as TensorRT becomes aware of the size they need to be.
 */
class CUDATensorPoolOutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  virtual ~CUDATensorPoolOutputAllocator() = default;
  explicit CUDATensorPoolOutputAllocator(exec::tensorrt::CUDATensorPool* pool) : pool_(pool) {}

  void notifyShape(const char*, const nvinfer1::Dims&) noexcept override;

  void* reallocateOutput(const char* tensor_name, void* current_memory, uint64_t size,
                         uint64_t alignment) noexcept override;

  StatusOr<exec::tensorrt::CUDATensorPtr> AcquireOutput(const std::string& name);
  const std::map<std::string, exec::tensorrt::CUDATensorPtr>& Outputs() { return outputs_; }
  void SetDataType(const std::string& name, exec::core::DataType data_type);

 private:
  exec::tensorrt::CUDATensorPool* pool_;
  std::map<std::string, exec::tensorrt::CUDATensorPtr> outputs_;
  std::map<std::string, exec::core::DataType> output_data_types_;
};

}  // namespace internal

/**
 * TensorRTExecuteCalculator Graph API:
 *  Input Side Packets:
 *   ExecutionContext tagged with EXEC_CTX
 *  Inputs:
 *   Each input must be a exec::tensorrt::CUDATensorPtr
 *  Outputs:
 *   Each output will be a exec::tensorrt::CUDATensorPtr
 **/
class TensorRTExecuteCalculator : public ExecutionContextBaseCalculator {
 public:
  using ExecutionContext = ::gml::gem::exec::tensorrt::ExecutionContext;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;
  Status ProcessImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;
  Status CloseImpl(mediapipe::CalculatorContext* cc, ExecutionContext* exec_ctx) override;

 private:
  std::unique_ptr<internal::CUDATensorPoolOutputAllocator> output_allocator_;
  optionspb::TensorRTExecuteCalculatorOptions options_;
};

}  // namespace gml::gem::calculators::tensorrt
