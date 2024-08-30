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

struct TestCase {
  std::map<std::string, std::string> assets;
};

class ModelBuilderTest : public ::testing::TestWithParam<TestCase> {};

TEST_P(ModelBuilderTest, BuildsWithoutError) {
  auto test_case = GetParam();

  ASSERT_OK_AND_ASSIGN(auto tmp_dir, fs::TempDir::Create());
  ASSERT_OK_AND_ASSIGN(auto blob_store, storage::FilesystemBlobStore::Create(tmp_dir->path()));

  std::filesystem::path testdata_path = "src/gem/build/plugin/openvino/testdata";

  ModelSpec spec;
  for (const auto& [name, path] : test_case.assets) {
    auto id = sole::uuid4();
    auto abs_path = bazel::RunfilePath(testdata_path / path);
    std::ifstream f(abs_path);
    std::string str_data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_OK(blob_store->Upsert(id.str(), str_data.c_str(), str_data.size()));

    auto* model_asset = spec.add_named_asset();
    model_asset->set_name(name);
    ToProto(id, model_asset->mutable_file()->mutable_file_id());
  }

  ModelBuilder builder;

  ASSERT_OK(builder.Build(blob_store.get(), spec));
}

INSTANTIATE_TEST_SUITE_P(ModelBuilderTestSuite, ModelBuilderTest,
                         ::testing::Values(
                             TestCase{
                                 {
                                     {"model", "simple.xml"},
                                     {"weight", "simple.bin"},
                                 },
                             },
                             TestCase{
                                 {
                                     {"model", "sharded.xml"},
                                     {"weight", "sharded.bin"},
                                     {"weights.shard0", "sharded.weights.shard0"},
                                 },
                             }));

}  // namespace gml::gem::build::openvino
