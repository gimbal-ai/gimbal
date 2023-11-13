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

#include "opentelemetry/exporters/otlp/otlp_metric_utils.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/metric_reader.h"

#include "src/common/base/base.h"

namespace gml::metrics {

/**
 * MetricsSystem is a wrapper around OTel metrics. It is structured such that it is
 * system-wide, with a single instance for the whole system.
 *
 * Creating stats:
 *
 * GetMeterProvider() can be used to create and add to new stats.
 * This pretty much gives direct access to OTel, so please follow OTel documentation for types
 * of stats and how to use them. There is a basic example included in the test for this file.
 *
 * Consuming stats:
 *
 * On the reader side, the CollectAllAsProto() can be used to extract all the stats in an OTel
 * defined protobuf, or Reader() can be called to set-up your own callback of how to process the
 * stats.
 *
 * The initial implementation of this metrics system is intentionally simple, avoid things like
 * OTel views. We may add these in the future, depending on the complexity of the stats collection.
 */
class MetricsSystem {
 public:
  /**
   * Get the instance to either add/update stats, or read stats out.
   */
  static MetricsSystem& GetInstance();

  /**
   * Main interface to creating and updating statistics. See OTel or test code for examples
   * of how to use it.
   */
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider> GetMeterProvider();

  /**
   * Returns all the stats in the system.
   */
  opentelemetry::proto::metrics::v1::ResourceMetrics CollectAllAsProto();

  /**
   * Returns the reader for custom access to the Collect() call.
   */
  opentelemetry::sdk::metrics::MetricReader* Reader();

 private:
  MetricsSystem() = default;

  static void Init();

  std::shared_ptr<opentelemetry::sdk::metrics::MetricReader> reader_;
};

}  // namespace gml::metrics
