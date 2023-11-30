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
