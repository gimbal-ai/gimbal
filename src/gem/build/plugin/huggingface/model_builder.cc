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

#include "src/gem/build/plugin/huggingface/model_builder.h"

#include <exception>

#include "src/common/base/error.h"
#include "src/common/uuid/uuid.h"
#include "src/gem/exec/plugin/huggingface/model.h"
#include "src/gem/exec/plugin/huggingface/tokenizer_wrapper/hf_fast_tokenizer.h"
#include "src/gem/storage/blob_store.h"

using ::gml::internal::api::core::v1::ModelSpec;

namespace gml::gem::build::huggingface {

StatusOr<std::unique_ptr<exec::core::Model>> ModelBuilder::Build(storage::BlobStore* store,
                                                                 const ModelSpec& spec) {
  bool model_found = false;
  types::UUID model_file_id;

  for (const auto& asset : spec.named_asset()) {
    if (asset.name() == "model") {
      model_file_id = asset.file().file_id();
      model_found = true;
    }
  }
  if (!model_found) {
    return error::InvalidArgument("HuggingFace tokenizer expects a json asset named 'model'");
  }
  GML_ASSIGN_OR_RETURN(auto model_path, store->FilePath(ParseUUID(model_file_id).str()));

  auto tokenizer = exec::HFFastTokenizer::Create(model_path);
  return std::unique_ptr<exec::core::Model>{
      new gml::gem::exec::huggingface::Model(std::move(tokenizer))};
}

}  // namespace gml::gem::build::huggingface
