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
#include "src/gem/exec/plugin/openvino/model.h"
#include "src/gem/storage/blob_store.h"

using ::gml::gem::exec::openvino::Model;
using ::gml::internal::api::core::v1::ModelSpec;

namespace gml::gem::build::openvino {

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(storage::BlobStore* store,
                                                                 const ModelSpec& spec) {
  GML_ASSIGN_OR_RETURN(auto onnx_path, store->FilePath(spec.onnx_blob_key()));

  ov::Core core;

  try {
    auto compiled_model = core.compile_model(onnx_path, "AUTO");
    return std::unique_ptr<exec::core::Model>{new Model(std::move(compiled_model))};
  } catch (std::exception e) {
    return error::Internal("Failed to compile openvino model: $0", e.what());
  }
}

}  // namespace gml::gem::build::openvino
