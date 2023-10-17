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

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "src/common/base/base.h"
#include "src/gem/core/exec/context.h"
#include "src/gem/plugins/tensorrt/exec/cuda_tensor_pool.h"

namespace gml {
namespace gem {
namespace tensorrt {

class TensorRTLogger : public nvinfer1::ILogger {
  void log(Severity, const char* msg) noexcept override {
    // For now just log everything as info. We should use the severity and look into using
    // tensorrt's error recorder infra.
    LOG(INFO) << "NvInfer log: " << msg;
  }
};

/**
 * ExecutionContext allows calculators to interact with a built TensorRT engine. It also has a
 * CUDATensorPool so calculators can allocate/recycle cuda tensors.
 */
class ExecutionContext : public core::ExecutionContext {
 public:
  ExecutionContext(TensorRTLogger&& logger, std::unique_ptr<nvinfer1::IRuntime> runtime,
                   std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine,
                   std::unique_ptr<nvinfer1::IExecutionContext> context)
      : logger_(std::move(logger)),
        tensor_pool_(),
        runtime_(std::move(runtime)),
        cuda_engine_(std::move(cuda_engine)),
        context_(std::move(context)) {}

  ~ExecutionContext() override {}

  nvinfer1::IExecutionContext* NVExecutionContext() { return context_.get(); }
  nvinfer1::ICudaEngine* CUDAEngine() { return cuda_engine_.get(); }
  CUDATensorPool* TensorPool() { return &tensor_pool_; }

  cudaStream_t CUDAStream();

 private:
  TensorRTLogger logger_;
  CUDATensorPool tensor_pool_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
