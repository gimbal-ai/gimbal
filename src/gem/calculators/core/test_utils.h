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

#include <string>

#include <absl/container/flat_hash_map.h>
#include <gmock/gmock.h>
#include <opentelemetry/sdk/common/attribute_utils.h>
#include <opentelemetry/sdk/metrics/meter.h>
#include <opentelemetry/sdk/metrics/sync_instruments.h>

#include "src/common/metrics/metrics_system.h"

struct ExpectedHist {
  std::vector<double> bucket_bounds;
  std::vector<uint64_t> bucket_counts;
  absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue> attributes =
      {};
};

auto OTelAttributes(
    const absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue>&
        expected_attributes) {
  using ::testing::Field;
  using ::testing::UnorderedElementsAreArray;

  namespace otel_metrics = opentelemetry::sdk::metrics;

  return Field(&otel_metrics::PointDataAttributes::attributes,
               UnorderedElementsAreArray(expected_attributes.begin(), expected_attributes.end()));
}

auto MatchHistogram(const ExpectedHist& expected) {
  using ::testing::AllOf;
  using ::testing::ElementsAreArray;
  using ::testing::Field;
  using ::testing::UnorderedElementsAreArray;
  using ::testing::VariantWith;

  namespace otel_metrics = opentelemetry::sdk::metrics;

  return AllOf(Field(&otel_metrics::PointDataAttributes::point_data,
                     VariantWith<otel_metrics::HistogramPointData>(
                         AllOf(Field(&otel_metrics::HistogramPointData::boundaries_,
                                     ElementsAreArray(expected.bucket_bounds)),
                               Field(&otel_metrics::HistogramPointData::counts_,
                                     ElementsAreArray(expected.bucket_counts))))),
               OTelAttributes(expected.attributes));
}

template <typename T>
struct ExpectedGaugeOrCounter {
  T value;
  absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue> attributes =
      {};
};

template <typename T, typename PointData>
auto MatchGaugeOrCounter(const ExpectedGaugeOrCounter<T>& expected) {
  using ::testing::AllOf;
  using ::testing::Field;
  using ::testing::UnorderedElementsAreArray;
  using ::testing::VariantWith;

  namespace otel_metrics = opentelemetry::sdk::metrics;

  return AllOf(Field(&otel_metrics::PointDataAttributes::point_data,
                     VariantWith<PointData>(Field(&PointData::value_, expected.value))),
               OTelAttributes(expected.attributes));
}
template <typename T>
auto MatchGauge(const ExpectedGaugeOrCounter<T>& expected) {
  return MatchGaugeOrCounter<T, opentelemetry::sdk::metrics::LastValuePointData>(expected);
}

template <typename T>
auto MatchCounter(const ExpectedGaugeOrCounter<T>& expected) {
  return MatchGaugeOrCounter<T, opentelemetry::sdk::metrics::SumPointData>(expected);
}

auto MatchHistogramVector(const std::vector<ExpectedHist>& expected) {
  using MatchHistogramType = decltype(MatchHistogram(expected[0]));

  std::vector<MatchHistogramType> matchers;
  matchers.reserve(expected.size());
  for (const auto& x : expected) {
    matchers.push_back(MatchHistogram(x));
  }
  return UnorderedElementsAreArray(matchers);
}

template <typename T>
struct ExpectedGauge : public ExpectedGaugeOrCounter<T> {};

template <typename T>
struct ExpectedCounter : public ExpectedGaugeOrCounter<T> {};

using ExpectedMetric = std::variant<ExpectedGauge<int64_t>, ExpectedCounter<int64_t>,
                                    ExpectedGauge<double>, ExpectedCounter<double>, ExpectedHist>;
void CheckMetrics(gml::metrics::MetricsSystem& metrics_system,
                  const absl::flat_hash_map<std::string, ExpectedMetric>& expected_metrics) {
  auto check_results =
      [&expected_metrics](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        const auto& scope_metrics = resource_metrics.scope_metric_data_;
        ASSERT_EQ(1, scope_metrics.size());

        const auto& metric_data = scope_metrics[0].metric_data_;
        ASSERT_EQ(expected_metrics.size(), metric_data.size());

        for (const auto& metric_datum : metric_data) {
          SCOPED_TRACE("For returned metric " + metric_datum.instrument_descriptor.name_);
          const auto& name = metric_datum.instrument_descriptor.name_;
          const auto& point_data = metric_datum.point_data_attr_;
          ASSERT_EQ(1, point_data.size());
          ASSERT_TRUE(expected_metrics.contains(name));
          const auto& expected_metric = expected_metrics.at(name);
          if (std::holds_alternative<ExpectedGauge<int64_t>>(expected_metric)) {
            EXPECT_THAT(point_data[0],
                        MatchGauge<int64_t>(std::get<ExpectedGauge<int64_t>>(expected_metric)));
          } else if (std::holds_alternative<ExpectedCounter<int64_t>>(expected_metric)) {
            EXPECT_THAT(point_data[0],
                        MatchCounter<int64_t>(std::get<ExpectedCounter<int64_t>>(expected_metric)));
          } else if (std::holds_alternative<ExpectedGauge<double>>(expected_metric)) {
            EXPECT_THAT(point_data[0],
                        MatchGauge<double>(std::get<ExpectedGauge<double>>(expected_metric)));
          } else if (std::holds_alternative<ExpectedCounter<double>>(expected_metric)) {
            EXPECT_THAT(point_data[0],
                        MatchCounter<double>(std::get<ExpectedCounter<double>>(expected_metric)));
          } else if (std::holds_alternative<ExpectedHist>(expected_metric)) {
            EXPECT_THAT(point_data[0], MatchHistogram(std::get<ExpectedHist>(expected_metric)));
          }
        }
      };

  auto results_cb =
      [&check_results](opentelemetry::sdk::metrics::ResourceMetrics& resource_metrics) {
        // Wrap this call because ASSERT_* macros return void.
        check_results(resource_metrics);
        return true;
      };
  metrics_system.Reader()->Collect(results_cb);
}
