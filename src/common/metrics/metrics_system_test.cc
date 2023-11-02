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

#include <google/protobuf/text_format.h>

#include "src/common/metrics/metrics_system.h"
#include "src/common/testing/testing.h"

using opentelemetry::proto::metrics::v1::ResourceMetrics;

TEST(MetricsTest, CollectAllAsProto) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();

  auto provider = metrics_sys.GetMeterProvider();
  auto meter = provider->GetMeter("gml-meter", "1.0.0");
  auto counter = meter->CreateUInt64Counter("counter");

  counter->Add(1);

  ResourceMetrics proto_resource_metrics = metrics_sys.CollectAllAsProto();

  ASSERT_EQ(proto_resource_metrics.scope_metrics_size(), 1);
  ASSERT_EQ(proto_resource_metrics.scope_metrics(0).metrics_size(), 1);
  EXPECT_EQ(proto_resource_metrics.scope_metrics(0).metrics(0).name(), "counter");
}

TEST(MetricsTest, Reader) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();

  auto provider = metrics_sys.GetMeterProvider();
  auto meter = provider->GetMeter("gml-meter", "1.0.0");
  auto counter = meter->CreateUInt64Counter("counter");

  counter->Add(1);

  auto reader = metrics_sys.Reader();

  int count = 0;
  auto results_cb = [&count](opentelemetry::sdk::metrics::ResourceMetrics& metric_data) {
    GML_UNUSED(metric_data);
    ++count;
    return true;
  };

  reader->Collect(results_cb);
  reader->Collect(results_cb);

  EXPECT_EQ(count, 2);
}
