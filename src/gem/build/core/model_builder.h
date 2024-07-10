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

#pragma once

#include "src/api/corepb/v1/model_exec.pb.h"
#include "src/common/base/base.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/storage/blob_store.h"

namespace gml::gem::build::core {

/**
 * ModelBuilder is the base class for plugin ModelBuilders.
 */
class ModelBuilder {
 public:
  virtual ~ModelBuilder() = default;
  virtual StatusOr<std::unique_ptr<::gml::gem::exec::core::Model>> Build(
      storage::BlobStore* store, const ::gml::internal::api::core::v1::ModelSpec& spec) = 0;
};

}  // namespace gml::gem::build::core
