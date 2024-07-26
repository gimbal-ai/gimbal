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

struct ExpectedHist {
  std::vector<double> bucket_bounds;
  std::vector<uint64_t> bucket_counts;
  absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue> attributes;
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
  absl::flat_hash_map<std::string, ::opentelemetry::sdk::common::OwnedAttributeValue> attributes;
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
