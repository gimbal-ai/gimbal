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

#include <openvino/openvino.hpp>

#include "src/common/base/error.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/exec/plugin/openvino/core_singleton.h"
#include "src/gem/exec/plugin/openvino/model.h"
#include "src/gem/storage/blob_store.h"

using ::gml::gem::exec::openvino::Model;
using ::gml::internal::api::core::v1::ModelSpec;

namespace gml::gem::build::openvino {

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(storage::BlobStore* store,
                                                                 const ModelSpec& spec) {
  bool model_found = false;
  types::UUID model_file_id;

  bool weights_found = false;
  types::UUID weights_file_id;

  for (const auto& asset : spec.named_asset()) {
    if (asset.name() == "weight") {
      weights_file_id = asset.file().file_id();
      weights_found = true;
    } else if (asset.name() == "model") {
      model_file_id = asset.file().file_id();
      model_found = true;
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
