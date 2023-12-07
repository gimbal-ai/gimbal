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
#include <opentelemetry/metrics/provider.h>

#include "src/common/metrics/metrics_system.h"
#include "src/common/system/proc_parser.h"

namespace gml::gem::controller {

class GEMMetricsReader : public gml::metrics::Scrapeable {
 public:
  GEMMetricsReader() = delete;
  explicit GEMMetricsReader(::gml::metrics::MetricsSystem* metrics_system);
  ~GEMMetricsReader() override = default;
  void Scrape() override;

 private:
  pid_t pgid_;
  std::unique_ptr<opentelemetry::metrics::UpDownCounter<int64_t>> cpu_counter_;
  std::unique_ptr<opentelemetry::metrics::UpDownCounter<int64_t>> mem_usage_counter_;
  std::unique_ptr<opentelemetry::metrics::UpDownCounter<int64_t>> mem_virtual_counter_;
  std::unique_ptr<opentelemetry::metrics::UpDownCounter<int64_t>> thread_counter_;
  std::unique_ptr<opentelemetry::metrics::UpDownCounter<int64_t>> context_switches_counter_;

  gml::system::ProcParser proc_parser_;
};

}  // namespace gml::gem::controller