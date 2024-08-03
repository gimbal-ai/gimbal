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

#include <filesystem>
#include <fstream>

#include <sole.hpp>

#include "src/common/bazel/runfiles.h"
#include "src/common/fs/temp_dir.h"
#include "src/common/testing/testing.h"
#include "src/common/uuid/uuid_utils.h"
#include "src/gem/storage/fs_blob_store.h"

namespace gml::gem::build::huggingface {

using ::gml::internal::api::core::v1::ModelSpec;

constexpr std::string_view kModelPath = "src/gem/build/plugin/huggingface/testdata/simple.json";

TEST(ModelBuilder, BuildsWithoutError) {
  auto model_path = bazel::RunfilePath(std::filesystem::path(kModelPath));
  ASSERT_OK_AND_ASSIGN(auto tmp_dir, fs::TempDir::Create());
  ASSERT_OK_AND_ASSIGN(auto blob_store, storage::FilesystemBlobStore::Create(tmp_dir->path()));

  auto model_asset_id = sole::uuid4();
  {
    std::ifstream f(model_path);
    std::string str_data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_OK(blob_store->Upsert(model_asset_id.str(), str_data.c_str(), str_data.size()));
  }

  ModelSpec spec;
  auto* model_asset = spec.add_named_asset();
  model_asset->set_name("model");
  ToProto(model_asset_id, model_asset->mutable_file()->mutable_file_id());

  ModelBuilder builder;

  ASSERT_OK(builder.Build(blob_store.get(), spec));
}

}  // namespace gml::gem::build::huggingface
