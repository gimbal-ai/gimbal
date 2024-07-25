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

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/substitute.h>
#include <glog/logging.h>
#include <opentelemetry/sdk/metrics/meter.h>

// We place the operator<< in the opentelemetry namespace to ensure that googletest matchers
// will be able to access it, as CPP requires them to be in the same namespace.
// https://google.github.io/googletest/advanced.html#teaching-googletest-how-to-print-your-values
namespace opentelemetry::sdk::metrics {

std::ostream& operator<<(std::ostream& os,
                         const opentelemetry::sdk::common::OwnedAttributeValue& value) {
  if (std::holds_alternative<bool>(value)) {
    os << (std::get<bool>(value) ? "true" : "false");
  } else if (std::holds_alternative<int>(value)) {
    os << std::to_string(std::get<int>(value));
  } else if (std::holds_alternative<int64_t>(value)) {
    os << std::to_string(std::get<int64_t>(value));
  } else if (std::holds_alternative<unsigned int>(value)) {
    os << std::to_string(std::get<unsigned int>(value));
  } else if (std::holds_alternative<uint64_t>(value)) {
    os << std::to_string(std::get<uint64_t>(value));
  } else if (std::holds_alternative<double>(value)) {
    os << std::to_string(std::get<double>(value));
  } else if (std::holds_alternative<std::string>(value)) {
    os << std::get<std::string>(value);
  } else {
    LOG(FATAL) << "Unexpected attribute value type encountered in AttributeValueToString";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const opentelemetry::sdk::common::OrderedAttributeMap& attrs) {
  os << "{";
  bool first = true;
  for (const auto& pair : attrs) {
    if (!first) {
      os << ", ";
    }
    os << pair.first << ": " << pair.second;
    first = false;
  }
  os << "}";
  return os;
}

template <typename T>
std::string PrintVector(const std::vector<T>& vec) {
  return absl::StrCat("[", absl::StrJoin(vec, ", "), "]");
}

std::ostream& operator<<(std::ostream& os,
                         const opentelemetry::sdk::metrics::HistogramPointData& hist) {
  os << "{"
     << "Boundaries: " << PrintVector(hist.boundaries_) << ", "
     << "Counts: " << PrintVector(hist.counts_) << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const opentelemetry::sdk::metrics::PointType& pt) {
  if (std::holds_alternative<opentelemetry::sdk::metrics::HistogramPointData>(pt)) {
    os << "Histogram: " << std::get<opentelemetry::sdk::metrics::HistogramPointData>(pt);
  } else {
    // TODO(philkuz) implement the other point types as they are needed.
    LOG(DFATAL) << "Unexpected point type encountered in PointTypeToString";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const opentelemetry::sdk::metrics::PointDataAttributes& pd) {
  os << "{" << pd.point_data << ", "
     << "Attributes: " << pd.attributes;
  os << "}";
  return os;
}
}  // namespace opentelemetry::sdk::metrics
