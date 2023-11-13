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

#include <absl/strings/str_split.h>
#include <google/protobuf/text_format.h>
#include <mediapipe/framework/calculator_graph.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>
#include "mediapipe/framework/calculator.pb.h"

#include "src/common/testing/testing.h"
#include "src/gem/exec/core/context.h"

namespace gml::gem::testing {

using ::gml::gem::exec::core::ExecutionContext;

class CalculatorTester : public mediapipe::CalculatorRunner {
 public:
  explicit CalculatorTester(const std::string& node_config)
      : mediapipe::CalculatorRunner(node_config), output_tag_map_(Outputs().TagMap()) {
    for (mediapipe::CollectionItemId id = Outputs().BeginId(); id < Outputs().EndId(); ++id) {
      packet_index_per_output_.emplace(id, 0);
    }
  }

  template <typename TData>
  CalculatorTester& ForInputSidePacket(int index, TData data) {
    return ForInputSidePacket<TData>("", index, std::move(data));
  }

  template <typename TData>
  CalculatorTester& ForInputSidePacket(std::string tag, TData data) {
    return ForInputSidePacket<TData>(tag, 0, std::move(data));
  }

  template <typename TData>
  CalculatorTester& ForInputSidePacket(const std::string& tag, int index, TData data) {
    MutableSidePackets()->Get(tag, index) = mediapipe::MakePacket<TData>(std::move(data));
    return *this;
  }

  template <typename TExecutionContext>
  CalculatorTester& WithExecutionContext(TExecutionContext* exec_ctx) {
    return ForInputSidePacket("EXEC_CTX", static_cast<ExecutionContext*>(exec_ctx));
  }

  template <typename TData>
  CalculatorTester& ForInput(const std::string& tag, TData data, int64_t timestamp) {
    return ForInput(tag, 0, std::move(data), timestamp);
  }

  template <typename TData>
  CalculatorTester& ForInput(int index, TData data, int64_t timestamp) {
    return ForInput("", index, std::move(data), timestamp);
  }

  template <typename TData>
  CalculatorTester& ForInput(const std::string& tag, int index, TData data, int64_t timestamp) {
    auto packet = mediapipe::MakePacket<TData>(std::move(data));
    packet = packet.At(mediapipe::Timestamp(timestamp));
    MutableInputs()->Get(tag, index).packets.emplace_back(std::move(packet));
    return *this;
  }

  CalculatorTester& Run() {
    EXPECT_OK(mediapipe::CalculatorRunner::Run());
    return *this;
  }

  template <typename TData, typename TMatcher>
  CalculatorTester& ExpectOutput(std::string tag, int64_t expected_timestamp, TMatcher matcher) {
    return ExpectOutput<TData>(tag, 0, expected_timestamp, matcher);
  }
  template <typename TData, typename TMatcher>
  CalculatorTester& ExpectOutput(size_t index, int64_t expected_timestamp, TMatcher matcher) {
    return ExpectOutput<TData>("", index, expected_timestamp, matcher);
  }

  template <typename TData, typename... TMatchers>
  CalculatorTester& ExpectOutput(const std::string& tag, int index, int64_t expected_timestamp,
                                 TMatchers... matchers) {
    auto item_id = output_tag_map_->GetId(tag, index);
    auto& packet_idx = packet_index_per_output_[item_id];

    const auto& packets = Outputs().Get(tag, index).packets;
    EXPECT_LT(packet_idx, packets.size());
    const auto& packet = packets.at(packet_idx);
    EXPECT_EQ(mediapipe::Timestamp(expected_timestamp), packet.Timestamp());
    EXPECT_THAT(packet.template Get<TData>(), ::testing::AllOf(matchers...));

    packet_idx++;
    return *this;
  }

  template <typename TData>
  const TData& Result(const std::string& tag, int index) {
    auto item_id = output_tag_map_->GetId(tag, index);
    auto& packet_idx = packet_index_per_output_[item_id];

    const auto& packets = Outputs().Get(tag, index).packets;
    EXPECT_LT(packet_idx, packets.size());
    const auto& packet = packets.at(packet_idx);

    packet_idx++;

    return packet.template Get<TData>();
  }

 private:
  std::shared_ptr<mediapipe::tool::TagMap> output_tag_map_;
  std::map<mediapipe::CollectionItemId, int> packet_index_per_output_;
};

}  // namespace gml::gem::testing
