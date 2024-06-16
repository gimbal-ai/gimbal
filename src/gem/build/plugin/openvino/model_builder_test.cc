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

#include <filesystem>
#include <fstream>

#include <sole.hpp>

#include "src/common/bazel/runfiles.h"
#include "src/common/fs/temp_dir.h"
#include "src/common/testing/testing.h"
#include "src/common/uuid/uuid_utils.h"
#include "src/gem/storage/fs_blob_store.h"

namespace gml::gem::build::openvino {

using ::gml::internal::api::core::v1::ModelSpec;

constexpr std::string_view kModelPath = "src/gem/build/plugin/openvino/testdata/simple.xml";
constexpr std::string_view kWeightPath = "src/gem/build/plugin/openvino/testdata/simple.bin";

TEST(ModelBuilder, BuildsWithoutError) {
  auto model_path = bazel::RunfilePath(std::filesystem::path(kModelPath));
  auto weight_path = bazel::RunfilePath(std::filesystem::path(kWeightPath));
  ASSERT_OK_AND_ASSIGN(auto tmp_dir, fs::TempDir::Create());
  ASSERT_OK_AND_ASSIGN(auto blob_store, storage::FilesystemBlobStore::Create(tmp_dir->path()));

  auto model_asset_id = sole::uuid4();
  {
    std::ifstream f(model_path);
    std::string str_data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_OK(blob_store->Upsert(model_asset_id.str(), str_data.c_str(), str_data.size()));
  }
  auto weight_asset_id = sole::uuid4();
  {
    std::ifstream f(weight_path);
    std::string str_data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_OK(blob_store->Upsert(weight_asset_id.str(), str_data.c_str(), str_data.size()));
  }

  ModelSpec spec;
  auto* model_asset = spec.add_named_asset();
  model_asset->set_name("model");
  ToProto(model_asset_id, model_asset->mutable_file()->mutable_file_id());
  auto* weight_asset = spec.add_named_asset();
  weight_asset->set_name("weight");
  ToProto(weight_asset_id, weight_asset->mutable_file()->mutable_file_id());

  ModelBuilder builder;

  ASSERT_OK(builder.Build(blob_store.get(), spec));
}

}  // namespace gml::gem::build::openvino
