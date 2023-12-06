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
