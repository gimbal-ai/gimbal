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
#include <NvOnnxParser.h>
#include <chrono>
#include "src/common/base/base.h"
#include "src/common/perf/elapsed_timer.h"

#include "src/gem/build/plugin/tensorrt/model_builder.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/exec/plugin/tensorrt/model.h"

namespace gml {
namespace gem {
namespace build {
namespace tensorrt {

using ::gml::gem::exec::tensorrt::Model;
using ::gml::gem::exec::tensorrt::TensorRTLogger;

StatusOr<std::unique_ptr<nvinfer1::IHostMemory>> BuildSerializedModel(storage::BlobStore* store,
                                                                      const specpb::ModelSpec& spec,
                                                                      TensorRTLogger logger) {
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  uint32_t flag =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

  {
    GML_ASSIGN_OR_RETURN(auto onnx_blob, store->MapReadOnly(spec.onnx_blob_key()));
    if (onnx_blob == nullptr) {
      return error::InvalidArgument("ONNX model not found in BlobStore with key $0",
                                    spec.onnx_blob_key());
    }
    parser->parse(onnx_blob->Data<char>(), onnx_blob->SizeForType<char>());
  }
  std::vector<std::string> errors;
  for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
    errors.emplace_back(parser->getError(i)->desc());
  }
  if (parser->getNbErrors() > 0) {
    return Status(gml::types::CODE_INVALID_ARGUMENT,
                  absl::StrCat("Failed to parse onnx file: ", absl::StrJoin(errors, ", ")));
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  for (const auto& opt_profile_spec : spec.tensorrt_spec().optimization_profile()) {
    auto* opt_profile = builder->createOptimizationProfile();
    for (const auto& tensor_shape_range : opt_profile_spec.tensor_shape_range()) {
      nvinfer1::Dims dims;
      dims.nbDims = tensor_shape_range.dim_size();
      for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = tensor_shape_range.dim(i);
      }
      opt_profile->setDimensions(tensor_shape_range.tensor_name().c_str(),
                                 nvinfer1::OptProfileSelector::kMIN, dims);
      opt_profile->setDimensions(tensor_shape_range.tensor_name().c_str(),
                                 nvinfer1::OptProfileSelector::kOPT, dims);
      opt_profile->setDimensions(tensor_shape_range.tensor_name().c_str(),
                                 nvinfer1::OptProfileSelector::kMAX, dims);
    }

    config->addOptimizationProfile(opt_profile);
  }

  auto serialized_model =
      std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  return serialized_model;
}

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(storage::BlobStore* store,
                                                                 const specpb::ModelSpec& spec) {
  ElapsedTimer timer;
  timer.Start();
  TensorRTLogger logger;

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine;

  auto engine_blob_key = spec.tensorrt_spec().engine_blob_key();
  if (engine_blob_key != "") {
    GML_ASSIGN_OR_RETURN(auto engine_blob, store->MapReadOnly(engine_blob_key));
    if (engine_blob != nullptr) {
      cuda_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(
          engine_blob->Data<char>(), engine_blob->SizeForType<char>()));
    }
  }
  if (cuda_engine == nullptr) {
    GML_ASSIGN_OR_RETURN(auto serialized_model, BuildSerializedModel(store, spec, logger));
    cuda_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size()));

    if (engine_blob_key != "") {
      GML_RETURN_IF_ERROR(store->Upsert(engine_blob_key,
                                        reinterpret_cast<const char*>(serialized_model->data()),
                                        serialized_model->size()));
    }
  }

  auto context =
      std::unique_ptr<nvinfer1::IExecutionContext>(cuda_engine->createExecutionContext());

  timer.Stop();

  auto elapsed_seconds = timer.ElapsedTime_us() / (1000 * 1000);

  LOG(INFO) << absl::Substitute("Successfully built TensorRT engine in $0s", elapsed_seconds);

  return std::unique_ptr<exec::core::Model>(
      new Model(std::move(logger), std::move(runtime), std::move(cuda_engine), std::move(context)));
}

}  // namespace tensorrt
}  // namespace build
}  // namespace gem
}  // namespace gml
