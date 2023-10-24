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

#include "src/common/base/base.h"
#include "src/gem/exec/core/model.h"
#include "src/gem/specpb/model.pb.h"

namespace gml {
namespace gem {
namespace build {
namespace core {

using ::gml::gem::exec::core::Model;

/**
 * ModelBuilder is the base class for plugin ModelBuilders.
 */
class ModelBuilder {
 public:
  virtual ~ModelBuilder() {}
  virtual StatusOr<std::unique_ptr<Model>> Build(const specpb::ModelSpec& spec) = 0;
};

}  // namespace core
}  // namespace build
}  // namespace gem
}  // namespace gml
