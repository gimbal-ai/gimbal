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

#include "src/gem/build/plugin/openvino/model_builder.h"

#include <exception>
#include <filesystem>
#include <fstream>

#include <openvino/openvino.hpp>

#include "src/common/base/error.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/fs/temp_dir.h"
#include "src/common/fs/temp_file.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/exec/plugin/openvino/core_singleton.h"
#include "src/gem/exec/plugin/openvino/model.h"
#include "src/gem/storage/blob_store.h"

using ::gml::gem::exec::openvino::Model;
using ::gml::internal::api::core::v1::ModelSpec;

namespace gml::gem::build::openvino {

namespace {
Status AppendFile(std::ostream* os, const std::filesystem::path& path) {
  GML_ASSIGN_OR_RETURN(auto stat, fs::Stat(path));
  if (stat.st_size == 0) {
    return Status::OK();
  }
  std::ifstream in(path);
  (*os) << in.rdbuf();
  return Status::OK();
}

StatusOr<std::string> ConcatenateWeights(const std::string& weights_path,
                                         const std::vector<std::string>& extra_weight_paths,
                                         const std::filesystem::path& output_dir) {
  // TODO(james): we need to rethink things to avoid having to concatenate the weight files
  // here.
  auto output_path = output_dir / "weights.bin";
  std::ofstream out(output_path);

  GML_RETURN_IF_ERROR(AppendFile(&out, weights_path));
  for (const auto& path : extra_weight_paths) {
    GML_RETURN_IF_ERROR(AppendFile(&out, path));
  }
  return output_path.string();
}
}  // namespace

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(storage::BlobStore* store,
                                                                 const ModelSpec& spec) {
  bool model_found = false;
  types::UUID model_file_id;

  bool weights_found = false;
  types::UUID weights_file_id;

  std::map<std::string, sole::uuid> extra_weight_shards;

  for (const auto& asset : spec.named_asset()) {
    if (asset.name() == "weight") {
      weights_file_id = asset.file().file_id();
      weights_found = true;
    } else if (asset.name() == "model") {
      model_file_id = asset.file().file_id();
      model_found = true;
    } else {
      extra_weight_shards[asset.name()] = ParseUUID(asset.file().file_id());
    }
  }
  if (!model_found) {
    return error::InvalidArgument("OpenVINO runtime expects a model asset named 'model'");
  }
  if (!weights_found) {
    return error::InvalidArgument("OpenVINO runtime expects a model asset named 'weight'");
  }

  GML_ASSIGN_OR_RETURN(auto model_path, store->FilePath(ParseUUID(model_file_id).str()));
  GML_ASSIGN_OR_RETURN(auto weights_path, store->FilePath(ParseUUID(weights_file_id).str()));
  GML_ASSIGN_OR_RETURN(auto weights_tmpdir, fs::TempDir::Create());

  if (!extra_weight_shards.empty()) {
    std::vector<std::string> extra_weight_paths;
    for (const auto& [_, shard_id] : extra_weight_shards) {
      GML_ASSIGN_OR_RETURN(auto path, store->FilePath(shard_id.str()));
      extra_weight_paths.push_back(path);
    }
    GML_ASSIGN_OR_RETURN(
        weights_path, ConcatenateWeights(weights_path, extra_weight_paths, weights_tmpdir->path()));
  }

  auto& core = exec::openvino::OpenVinoCoreGetInstance();

  try {
    // For now explicitly select between GPU and CPU based on availability.
    // TODO(james): investigate why 'AUTO' plugin doesn't pick the GPU when available.
    auto available_devices = core.get_available_devices();
    std::string device = "CPU";

    // TODO(james): re-enable the GPU once the compiler issues are fixed.
    for (const auto& dev : available_devices) {
      if (absl::StartsWith(dev, "GPU")) {
        device = "GPU";
      }
    }

    LOG(INFO) << absl::Substitute("Using $0 to execute $1", device, spec.name());

    auto model = core.read_model(model_path, weights_path);

    // TODO(james): convert models to fp16 in the compiler and remove the inference precision hint
    // here.
    auto compiled_model = core.compile_model(model, device, ov::hint::inference_precision("f32"));
    return std::unique_ptr<exec::core::Model>{new Model(std::move(compiled_model))};
  } catch (const std::exception& e) {
    return error::Internal("Failed to compile openvino model: $0", e.what());
  }
}

}  // namespace gml::gem::build::openvino
