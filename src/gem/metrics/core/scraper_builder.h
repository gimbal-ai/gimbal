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
#include "src/common/metrics/metrics_system.h"

namespace gml::gem::metrics::core {

using Scraper = gml::metrics::Scrapeable;

/**
 * ScraperBuilder is the base class for plugin ScraperBuilders.
 */
class ScraperBuilder {
 public:
  virtual ~ScraperBuilder() = default;
  virtual StatusOr<std::unique_ptr<Scraper>> Build(gml::metrics::MetricsSystem*) = 0;
};

/**
 * DefaultScraperBuilder is a convenience class for Scrapers that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TScraper>
class DefaultScraperBuilder : public ScraperBuilder {
 public:
  ~DefaultScraperBuilder() override = default;

  StatusOr<std::unique_ptr<Scraper>> Build(gml::metrics::MetricsSystem* metrics_system) override {
    return BuildInternal(metrics_system);
  }

 private:
  template <typename Q = TScraper,
            std::enable_if_t<std::is_constructible_v<Q, gml::metrics::MetricsSystem*>>* = nullptr>
  StatusOr<std::unique_ptr<Scraper>> BuildInternal(gml::metrics::MetricsSystem* metrics_system) {
    return std::unique_ptr<Scraper>(new Q(metrics_system));
  }

  template <typename Q = TScraper,
            std::enable_if_t<!std::is_constructible_v<Q, gml::metrics::MetricsSystem*>>* = nullptr>
  StatusOr<std::unique_ptr<Scraper>> BuildInternal() {
    static_assert(sizeof(Q) == 0,
                  "Scraper must have a single argument constructor with "
                  "gml::metrics::MetricsSystem* as the argument");
    return Status();
  }
};

}  // namespace gml::gem::metrics::core
