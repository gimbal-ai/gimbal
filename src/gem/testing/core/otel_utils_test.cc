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

#include "src/gem/testing/core/otel_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opentelemetry/sdk/common/attribute_utils.h>
#include <opentelemetry/sdk/metrics/data/point_data.h>

namespace opentelemetry::sdk::metrics {
namespace {

TEST(OtelSdkMetricsUtilsTest, AttributeValueToString) {
  std::ostringstream oss;

  oss << opentelemetry::sdk::common::OwnedAttributeValue(true);
  EXPECT_EQ(oss.str(), "true");
  oss.str("");

  oss << opentelemetry::sdk::common::OwnedAttributeValue(false);
  EXPECT_EQ(oss.str(), "false");
  oss.str("");

  oss << opentelemetry::sdk::common::OwnedAttributeValue(42);
  EXPECT_EQ(oss.str(), "42");
  oss.str("");

  oss << opentelemetry::sdk::common::OwnedAttributeValue(int64_t{1234567890});
  EXPECT_EQ(oss.str(), "1234567890");
  oss.str("");

  oss << opentelemetry::sdk::common::OwnedAttributeValue(3.14);
  EXPECT_EQ(oss.str(), "3.140000");
  oss.str("");

  oss << opentelemetry::sdk::common::OwnedAttributeValue(std::string("hello"));
  EXPECT_EQ(oss.str(), "hello");
}

TEST(OtelSdkMetricsUtilsTest, PrintVector) {
  std::vector<int> int_vec = {1, 2, 3, 4, 5};
  EXPECT_EQ(PrintVector(int_vec), "[1, 2, 3, 4, 5]");

  std::vector<std::string> str_vec = {"a", "b", "c"};
  EXPECT_EQ(PrintVector(str_vec), "[a, b, c]");
}

TEST(OtelSdkMetricsUtilsTest, OrderedAttributeMapToString) {
  opentelemetry::sdk::common::OrderedAttributeMap attrs = {
      {"key1", "value1"}, {"key2", 42}, {"key3", true}};
  std::ostringstream oss;
  oss << attrs;
  EXPECT_EQ(oss.str(), "{key1: value1, key2: 42, key3: true}");
}

TEST(OtelSdkMetricsUtilsTest, HistogramToString) {
  HistogramPointData hist;
  hist.boundaries_ = {0.0, 5.0, 10.0};
  hist.counts_ = {1, 2, 3, 4};

  std::ostringstream oss;
  oss << hist;
  EXPECT_EQ(oss.str(), "{Boundaries: [0, 5, 10], Counts: [1, 2, 3, 4]}");

  PointType pt = hist;
  oss.str("");  // Clear the stream
  oss << pt;
  EXPECT_EQ(oss.str(), "Histogram: {Boundaries: [0, 5, 10], Counts: [1, 2, 3, 4]}");
}

TEST(OtelSdkMetricsUtilsTest, PointDataAttributesToString) {
  HistogramPointData hist;
  hist.boundaries_ = {0.0, 5.0, 10.0};
  hist.counts_ = {1, 2, 3, 4};

  opentelemetry::sdk::common::OrderedAttributeMap attrs = {{"key1", "value1"}, {"key2", 42}};

  PointDataAttributes pd{attrs, hist};

  std::ostringstream oss;
  oss << pd;
  EXPECT_EQ(oss.str(),
            "{Histogram: {Boundaries: [0, 5, 10], Counts: [1, 2, 3, 4]}, Attributes: "
            "{key1: value1, key2: 42}}");
}

}  // namespace
}  // namespace opentelemetry::sdk::metrics
