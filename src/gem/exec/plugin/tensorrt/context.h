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

#pragma once

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/exec/plugin/tensorrt/cuda_tensor_pool.h"
#include "src/gem/exec/plugin/tensorrt/model.h"

namespace gml::gem::exec::tensorrt {

/**
 * ExecutionContext allows calculators to interact with a built TensorRT model. It also has a
 * CUDATensorPool so calculators can allocate/recycle cuda tensors.
 */
class ExecutionContext : public core::ExecutionContext {
 public:
  explicit ExecutionContext(core::Model* model) : model_(static_cast<tensorrt::Model*>(model)) {}

  ~ExecutionContext() override = default;

  nvinfer1::IExecutionContext* NVExecutionContext() { return model_->NVExecutionContext(); }
  nvinfer1::ICudaEngine* CUDAEngine() { return model_->CUDAEngine(); }
  CUDATensorPool* TensorPool() { return &tensor_pool_; }

  cudaStream_t CUDAStream();

 private:
  CUDATensorPool tensor_pool_;
  Model* model_;
};

}  // namespace gml::gem::exec::tensorrt
