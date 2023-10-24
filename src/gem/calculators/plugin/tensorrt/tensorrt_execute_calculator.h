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

#include <NvInfer.h>

#include "src/gem/calculators/core/execution_context_calculator.h"
#include "src/gem/calculators/plugin/tensorrt/base.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace calculators {
namespace tensorrt {

using ::gml::gem::exec::tensorrt::CUDATensorPool;
using ::gml::gem::exec::tensorrt::CUDATensorPtr;

namespace internal {
/**
 * CUDATensorPoolOutputAllocator implements TensorRT's IOutputAllocator to dynamically allocate
 * CUDATensors as TensorRT becomes aware of the size they need to be.
 */
class CUDATensorPoolOutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  virtual ~CUDATensorPoolOutputAllocator() {}
  explicit CUDATensorPoolOutputAllocator(CUDATensorPool* pool) : pool_(pool) {}

  void notifyShape(const char*, const nvinfer1::Dims&) noexcept override;

  void* reallocateOutput(const char* tensor_name, void* current_memory, uint64_t size,
                         uint64_t alignment) noexcept override;

  StatusOr<CUDATensorPtr> AcquireOutput(const std::string& name);
  const std::map<std::string, CUDATensorPtr>& Outputs() { return outputs_; }

 private:
  CUDATensorPool* pool_;
  std::map<std::string, CUDATensorPtr> outputs_;
  std::map<std::string, nvinfer1::Dims> shapes_;
};

}  // namespace internal

/**
 * TensorRTExecuteCalculator Graph API:
 *  Input Side Packets:
 *   ExecutionContext tagged with EXEC_CTX
 *  Inputs:
 *   Each input must be a CUDATensorPtr
 *  Outputs:
 *   Each output will be a CUDATensorPtr
 **/
class TensorRTExecuteCalculator : public ExecutionContextBaseCalculator {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc, tensorrt::ExecutionContext* exec_ctx) override;
  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     tensorrt::ExecutionContext* exec_ctx) override;
  Status CloseImpl(mediapipe::CalculatorContext* cc, tensorrt::ExecutionContext* exec_ctx) override;

 private:
  std::unique_ptr<internal::CUDATensorPoolOutputAllocator> output_allocator_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace tensorrt
}  // namespace calculators
}  // namespace gem
}  // namespace gml
