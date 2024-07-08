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

#include "src/common/metrics/metrics_system.h"

#include <google/protobuf/text_format.h>

#include "src/common/testing/testing.h"

using opentelemetry::proto::metrics::v1::ResourceMetrics;

TEST(MetricsTest, CollectAllAsProto) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();
  metrics_sys.Reset();

  auto provider = metrics_sys.GetMeterProvider();
  auto meter = provider->GetMeter("gml-meter", "1.0.0");
  auto uint64_counter = meter->CreateUInt64Counter("uint64_counter");
  auto double_counter = meter->CreateDoubleCounter("double_counter");
  auto int64_gauge = meter->CreateInt64Gauge("int64_gauge");
  auto double_gauge = meter->CreateDoubleGauge("double_gauge");

  uint64_counter->Add(1);
  double_counter->Add(11.1);
  int64_gauge->Record(2);
  double_gauge->Record(22.2);

  ResourceMetrics proto_resource_metrics = metrics_sys.CollectAllAsProto();

  ASSERT_EQ(proto_resource_metrics.scope_metrics_size(), 1);
  ASSERT_EQ(proto_resource_metrics.scope_metrics(0).metrics_size(), 4);

  std::vector<std::string> names;
  for (const auto& metric : proto_resource_metrics.scope_metrics(0).metrics()) {
    names.push_back(metric.name());
  }
  ASSERT_THAT(names, ::testing::UnorderedElementsAre("uint64_counter", "double_counter",
                                                     "int64_gauge", "double_gauge"));
}

TEST(MetricsTest, ChunkMetrics) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();
  metrics_sys.Reset();

  auto provider = metrics_sys.GetMeterProvider();
  auto meter = provider->GetMeter("gml-meter", "1.0.0");
  auto uint64_counter = meter->CreateUInt64Counter("uint64_counter");
  auto double_counter = meter->CreateDoubleCounter("double_counter");
  auto int64_gauge = meter->CreateInt64Gauge("int64_gauge");
  auto double_gauge = meter->CreateDoubleGauge("double_gauge");

  uint64_counter->Add(1);
  double_counter->Add(11.1);
  int64_gauge->Record(2);
  double_gauge->Record(22.2);

  auto meter2 = provider->GetMeter("gml-meter2", "1.0.0");
  auto uint64_counter2 = meter2->CreateUInt64Counter("uint64_counter2");
  auto double_counter2 = meter2->CreateDoubleCounter("double_counter2");
  auto int64_gauge2 = meter2->CreateInt64Gauge("int64_gauge2");
  auto double_gauge2 = meter2->CreateDoubleGauge("double_gauge2");
  auto double_gauge3 = meter2->CreateDoubleGauge("double_gauge3");

  uint64_counter2->Add(1);
  double_counter2->Add(11.1);
  int64_gauge2->Record(2);
  double_gauge2->Record(22.2);
  double_gauge3->Record(33.3);

  ResourceMetrics resource_metrics = metrics_sys.CollectAllAsProto();

  auto proto_resource_metrics = metrics_sys.ChunkMetrics(&resource_metrics, 300);

  ASSERT_EQ(proto_resource_metrics.size(), 5);

  std::vector<std::string> names;
  int64_t num_metrics = 0;
  int64_t num_scope_metrics = 0;
  for (const auto& resource_metrics : proto_resource_metrics) {
    for (const auto& scope_metric : resource_metrics.scope_metrics()) {
      for (const auto& metric : scope_metric.metrics()) {
        names.push_back(metric.name());
        num_metrics++;
      }
    }
    num_scope_metrics++;
  }

  ASSERT_EQ(num_scope_metrics, 5);
  ASSERT_EQ(num_metrics, 9);
  ASSERT_THAT(names,
              ::testing::UnorderedElementsAre("uint64_counter", "double_counter", "int64_gauge",
                                              "double_gauge", "uint64_counter2", "double_counter2",
                                              "int64_gauge2", "double_gauge2", "double_gauge3"));
}

TEST(MetricsTest, Reader) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();
  metrics_sys.Reset();

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

TEST(MetricsTest, CreateHistogramWithBounds) {
  ::gml::metrics::MetricsSystem& metrics_sys = ::gml::metrics::MetricsSystem::GetInstance();
  metrics_sys.Reset();

  auto hist = metrics_sys.CreateHistogramWithBounds<uint64_t>("test_metric", "this is my metric",
                                                              {0.0, 10.0, 50.0});

  hist->Record(1, {{"test", "test1"}}, {});
  hist->Record(10, {{"test", "test1"}}, {});
  hist->Record(11, {{"test", "test2"}}, {});

  ResourceMetrics proto = metrics_sys.CollectAllAsProto();

  ASSERT_EQ(1, proto.scope_metrics_size());
  ASSERT_EQ(1, proto.scope_metrics(0).metrics_size());
  auto metric = proto.scope_metrics(0).metrics(0);
  EXPECT_EQ(metric.name(), "test_metric");
  ASSERT_TRUE(metric.has_histogram());
  const auto& histogram = metric.histogram();
  ASSERT_EQ(2, histogram.data_points_size());
  const auto& dp0 = histogram.data_points(0);
  ASSERT_EQ(3, dp0.explicit_bounds_size());
  EXPECT_THAT(dp0.explicit_bounds(), ::testing::ElementsAre(0.0, 10.0, 50.0));
  ASSERT_EQ(4, dp0.bucket_counts_size());
  EXPECT_THAT(dp0.bucket_counts(), ::testing::ElementsAre(0, 2, 0, 0));

  const auto& dp1 = histogram.data_points(1);
  ASSERT_EQ(3, dp1.explicit_bounds_size());
  EXPECT_THAT(dp1.explicit_bounds(), ::testing::ElementsAre(0.0, 10.0, 50.0));
  ASSERT_EQ(4, dp1.bucket_counts_size());
  EXPECT_THAT(dp1.bucket_counts(), ::testing::ElementsAre(0, 0, 1, 0));
}
