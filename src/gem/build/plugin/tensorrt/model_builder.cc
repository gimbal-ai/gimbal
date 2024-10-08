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

#include "src/gem/build/plugin/tensorrt/model_builder.h"

#include <chrono>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "src/common/base/base.h"
#include "src/common/base/error.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/fs/temp_dir.h"
#include "src/common/perf/elapsed_timer.h"
#include "src/common/system/memory_mapped_file.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/exec/plugin/tensorrt/model.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::build::tensorrt {

using ::gml::gem::exec::tensorrt::Model;
using ::gml::gem::exec::tensorrt::TensorRTLogger;
using ::gml::internal::api::core::v1::ModelSpec;

namespace {

// TODO(james): we should change empty string to "model" across the whole system.
constexpr std::string_view kModelAssetName = "";
constexpr std::string_view kModelFileName = "model.onnx";

Status ParseFromAssets(storage::BlobStore* store, const ModelSpec& spec,
                       nvonnxparser::IParser* parser) {
  // Because of the way onnx handles external data, we have to build a symlink dir where each file
  // is named after the asset name.
  GML_ASSIGN_OR_RETURN(auto tmpdir, fs::TempDir::Create());

  for (const auto& asset : spec.named_asset()) {
    auto file_id = ParseUUID(asset.file().file_id()).str();
    GML_ASSIGN_OR_RETURN(auto path, store->FilePath(file_id));
    std::string new_name;
    if (asset.name() == kModelAssetName) {
      new_name = kModelFileName;
    } else {
      new_name = asset.name();
    }

    GML_RETURN_IF_ERROR(fs::CreateSymlink(path, tmpdir->path() / new_name));
  }

  auto model_path = tmpdir->path() / kModelFileName;
  parser->parseFromFile(model_path.string().c_str(), /*verbosity*/ 0);

  std::vector<std::string> errors;
  errors.reserve(parser->getNbErrors());
  for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
    errors.emplace_back(parser->getError(i)->desc());
  }
  if (parser->getNbErrors() > 0) {
    return error::InvalidArgument("Failed to parse onnx file: $0", absl::StrJoin(errors, ", "));
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<nvinfer1::IHostMemory>> BuildSerializedModel(storage::BlobStore* store,
                                                                      const ModelSpec& spec,
                                                                      TensorRTLogger logger) {
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  uint32_t flag =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

  GML_RETURN_IF_ERROR(ParseFromAssets(store, spec, parser.get()));

  bool needs_int8 = false;
  for (int i = 0; i < network->getNbLayers(); ++i) {
    auto* layer = network->getLayer(i);
    if (layer->getType() == nvinfer1::LayerType::kDEQUANTIZE ||
        layer->getType() == nvinfer1::LayerType::kQUANTIZE) {
      needs_int8 = true;
      break;
    }
  }
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (needs_int8) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }

  // Set the optimization profiles to have the shapes of the inputs. The compiler currently enforces
  // that these are static.
  auto* opt_profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto* input = network->getInput(i);
    opt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                               input->getDimensions());
    opt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                               input->getDimensions());
    opt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                               input->getDimensions());
    if (input->isShapeTensor()) {
      // If there is a shape input, assume its the shape of an image and set hardcoded image size
      // bounds.
      // TODO(james): We should set this in the model spec and pass it down from the compiler.
      std::array<int32_t, 2> min_values = {1, 1};
      std::array<int32_t, 2> opt_values = {640, 480};
      std::array<int32_t, 2> max_values = {4096, 4096};
      opt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                  min_values.data(), min_values.size());
      opt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                  opt_values.data(), opt_values.size());
      opt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                  max_values.data(), max_values.size());
    }
  }
  config->addOptimizationProfile(opt_profile);
  if (spec.tensorrt_spec().mem_pool_limits().workspace() > 0) {
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               spec.tensorrt_spec().mem_pool_limits().workspace());
  }

  auto serialized_model =
      std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  return serialized_model;
}

}  // namespace

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(
    storage::BlobStore* store, const ::gml::internal::api::core::v1::ModelSpec& spec) {
  ElapsedTimer timer;
  timer.Start();
  TensorRTLogger logger;

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine;

  auto engine_blob_key = spec.tensorrt_spec().engine_blob_key();
  if (engine_blob_key != "") {
    auto engine_path_or_s = store->FilePath(engine_blob_key);
    if (engine_path_or_s.ok()) {
      GML_ASSIGN_OR_RETURN(auto engine_mmap,
                           system::MemoryMappedFile::MapReadOnly(engine_path_or_s.ValueOrDie()));
      const auto* engine_data = reinterpret_cast<const char*>(engine_mmap->data());
      cuda_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(engine_data, engine_mmap->size()));
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

  auto elapsed_seconds = timer.ElapsedTime_us() / (1000ULL * 1000ULL);

  LOG(INFO) << absl::Substitute("Successfully built TensorRT engine in $0s", elapsed_seconds);

  return std::unique_ptr<exec::core::Model>(
      new Model(std::move(logger), std::move(runtime), std::move(cuda_engine), std::move(context)));
}

}  // namespace gml::gem::build::tensorrt
