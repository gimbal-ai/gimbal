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
#include "src/gem/core/exec/model.h"
#include "src/gem/plugins/tensorrt/exec/cuda_tensor_pool.h"
#include "src/gem/plugins/tensorrt/exec/model.h"

namespace gml {
namespace gem {
namespace tensorrt {

/**
 * ExecutionContext allows calculators to interact with a built TensorRT model. It also has a
 * CUDATensorPool so calculators can allocate/recycle cuda tensors.
 */
class ExecutionContext : public core::ExecutionContext {
 public:
  ExecutionContext(core::Model* model) : model_(static_cast<tensorrt::Model*>(model)) {}

  ~ExecutionContext() override {}

  nvinfer1::IExecutionContext* NVExecutionContext() { return model_->NVExecutionContext(); }
  nvinfer1::ICudaEngine* CUDAEngine() { return model_->CUDAEngine(); }
  CUDATensorPool* TensorPool() { return &tensor_pool_; }

  cudaStream_t CUDAStream();

 private:
  CUDATensorPool tensor_pool_;
  Model* model_;
};

}  // namespace tensorrt
}  // namespace gem
}  // namespace gml
