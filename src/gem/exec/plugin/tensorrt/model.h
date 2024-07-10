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
#include "src/gem/exec/core/model.h"

namespace gml::gem::exec::tensorrt {

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
class Model : public core::Model {
 public:
  Model(TensorRTLogger&& logger, std::unique_ptr<nvinfer1::IRuntime> runtime,
        std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine,
        std::unique_ptr<nvinfer1::IExecutionContext> context)
      : logger_(std::move(logger)),
        runtime_(std::move(runtime)),
        cuda_engine_(std::move(cuda_engine)),
        context_(std::move(context)) {}

  ~Model() override = default;
  nvinfer1::IExecutionContext* NVExecutionContext() { return context_.get(); }
  nvinfer1::ICudaEngine* CUDAEngine() { return cuda_engine_.get(); }

 private:
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace gml::gem::exec::tensorrt
