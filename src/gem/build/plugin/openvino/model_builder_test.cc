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

#include "src/common/testing/test_environment.h"
#include "src/common/testing/testing.h"
#include "src/gem/storage/fs_blob_store.h"

namespace gml::gem::build::openvino {

using ::gml::internal::api::core::v1::ModelSpec;

constexpr std::string_view kOnnxPath = "src/gem/build/plugin/openvino/testdata/simple.onnx";

TEST(ModelBuilder, BuildsWithoutError) {
  auto onnx_path = testing::BazelRunfilePath(std::filesystem::path(kOnnxPath));
  ASSERT_OK_AND_ASSIGN(auto blob_store,
                       storage::FilesystemBlobStore::Create(onnx_path.parent_path()));

  ModelSpec spec;
  spec.set_onnx_blob_key(onnx_path.filename());

  ModelBuilder builder;

  ASSERT_OK(builder.Build(blob_store.get(), spec));
}

}  // namespace gml::gem::build::openvino
