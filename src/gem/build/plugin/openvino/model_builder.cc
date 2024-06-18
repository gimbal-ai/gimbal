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
    // for (const auto& dev : available_devices) {
    //  if (absl::StartsWith(dev, "GPU")) {
    //    device = "GPU";
    //  }
    //}

    LOG(INFO) << absl::Substitute("Using $0 to execute $1", device, spec.name());

    auto model = core.read_model(model_path, weights_path);

    auto compiled_model = core.compile_model(model, device);
    return std::unique_ptr<exec::core::Model>{new Model(std::move(compiled_model))};
  } catch (const std::exception& e) {
    return error::Internal("Failed to compile openvino model: $0", e.what());
  }
}

}  // namespace gml::gem::build::openvino
