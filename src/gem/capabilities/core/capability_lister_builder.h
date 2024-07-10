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

#include "src/common/base/base.h"
#include "src/gem/capabilities/core/capability_lister.h"

namespace gml::gem::capabilities::core {

/**
 * CapabilityListerBuilder is the base class for plugin CapabilityListerBuilders.
 */
class CapabilityListerBuilder {
 public:
  using CapabilityLister = ::gml::gem::capabilities::core::CapabilityLister;

  virtual ~CapabilityListerBuilder() = default;
  virtual StatusOr<std::unique_ptr<CapabilityLister>> Build() = 0;
};

/**
 * DefaultCapabilityListerBuilder is a convenience class for CapabilityListers that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TCapabilityLister>
class DefaultCapabilityListerBuilder : public CapabilityListerBuilder {
 public:
  ~DefaultCapabilityListerBuilder() override = default;

  StatusOr<std::unique_ptr<CapabilityLister>> Build() override { return BuildInternal(); }

 private:
  template <typename Q = TCapabilityLister, std::enable_if_t<std::is_constructible_v<Q>>* = nullptr>
  StatusOr<std::unique_ptr<CapabilityLister>> BuildInternal() {
    return std::unique_ptr<CapabilityLister>(new Q);
  }

  template <typename Q = TCapabilityLister,
            std::enable_if_t<!std::is_constructible_v<Q>>* = nullptr>
  StatusOr<std::unique_ptr<CapabilityLister>> BuildInternal() {
    static_assert(sizeof(Q) == 0, "CapabilityLister must have default constructor");
    return Status();
  }
};

}  // namespace gml::gem::capabilities::core
