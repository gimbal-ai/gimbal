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

#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/gem/metrics/core/scraper_builder.h"

namespace gml::gem::metrics::openvino {
class OpenVinoGPUMetrics : public core::Scraper {
 public:
  explicit OpenVinoGPUMetrics(gml::metrics::MetricsSystem* metrics_system,
                              gml::event::Dispatcher* dispatcher);

  void Scrape() override;

 private:
  std::unique_ptr<opentelemetry::metrics::Gauge<uint64_t>> gpu_mem_usage_gauge_;
};
}  // namespace gml::gem::metrics::openvino
