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
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"

namespace gml::gem::metrics::core {

using Scraper = gml::metrics::Scrapeable;

/**
 * ScraperBuilder is the base class for plugin ScraperBuilders.
 */
class ScraperBuilder {
 public:
  virtual ~ScraperBuilder() = default;
  virtual StatusOr<std::unique_ptr<Scraper>> Build(gml::metrics::MetricsSystem*,
                                                   gml::event::Dispatcher* dispatcher) = 0;
};

/**
 * DefaultScraperBuilder is a convenience class for Scrapers that don't need a
 * builder and can just be created with std::make_unique.
 */
template <typename TScraper>
class DefaultScraperBuilder : public ScraperBuilder {
 public:
  ~DefaultScraperBuilder() override = default;

  StatusOr<std::unique_ptr<Scraper>> Build(gml::metrics::MetricsSystem* metrics_system,
                                           gml::event::Dispatcher* dispatcher) override {
    return BuildInternal(metrics_system, dispatcher);
  }

 private:
  template <typename Q = TScraper,
            std::enable_if_t<std::is_constructible_v<Q, gml::metrics::MetricsSystem*,
                                                     gml::event::Dispatcher*>>* = nullptr>
  StatusOr<std::unique_ptr<Scraper>> BuildInternal(gml::metrics::MetricsSystem* metrics_system,
                                                   gml::event::Dispatcher* dispatcher) {
    return std::unique_ptr<Scraper>(new Q(metrics_system, dispatcher));
  }

  template <typename Q = TScraper,
            std::enable_if_t<!std::is_constructible_v<Q, gml::metrics::MetricsSystem*,
                                                      gml::event::Dispatcher*>>* = nullptr>
  StatusOr<std::unique_ptr<Scraper>> BuildInternal() {
    static_assert(sizeof(Q) == 0,
                  "Scraper must have constructor with "
                  "gml::metrics::MetricsSystem* and gml::event::Dispatcher* as arguments");
    return Status();
  }
};

}  // namespace gml::gem::metrics::core
