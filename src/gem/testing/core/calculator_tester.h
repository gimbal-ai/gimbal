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

#include <absl/strings/str_split.h>
#include <google/protobuf/text_format.h>
#include <mediapipe/framework/calculator.pb.h>
#include <mediapipe/framework/calculator_graph.h>
#include <mediapipe/framework/calculator_runner.h>
#include <mediapipe/framework/packet.h>

#include "src/common/testing/testing.h"
#include "src/gem/exec/core/context.h"

namespace gml::gem::testing {

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
    using ::gml::gem::exec::core::ExecutionContext;
    return ForInputSidePacket("EXEC_CTX", static_cast<ExecutionContext*>(exec_ctx));
  }

  template <typename TData>
  CalculatorTester& ForInput(const std::string& tag, TData data, mediapipe::Timestamp timestamp) {
    return ForInput(tag, 0, std::move(data), timestamp);
  }

  template <typename TData>
  CalculatorTester& ForInput(int index, TData data, mediapipe::Timestamp timestamp) {
    return ForInput("", index, std::move(data), timestamp);
  }

  template <typename TData>
  CalculatorTester& ForInput(const std::string& tag, int index, TData data,
                             mediapipe::Timestamp timestamp) {
    auto packet = mediapipe::MakePacket<TData>(std::move(data));
    packet = packet.At(timestamp);
    MutableInputs()->Get(tag, index).packets.emplace_back(std::move(packet));
    return *this;
  }

  CalculatorTester& Run() {
    EXPECT_OK(mediapipe::CalculatorRunner::Run());
    return *this;
  }

  template <typename TData, typename TMatcher>
  CalculatorTester& ExpectOutput(std::string tag, mediapipe::Timestamp expected_timestamp,
                                 TMatcher matcher) {
    return ExpectOutput<TData>(tag, 0, expected_timestamp, matcher);
  }
  template <typename TData, typename TMatcher>
  CalculatorTester& ExpectOutput(size_t index, mediapipe::Timestamp expected_timestamp,
                                 TMatcher matcher) {
    return ExpectOutput<TData>("", index, expected_timestamp, matcher);
  }

  template <typename TData, typename... TMatchers>
  CalculatorTester& ExpectOutput(const std::string& tag, int index,
                                 mediapipe::Timestamp expected_timestamp, TMatchers... matchers) {
    auto item_id = output_tag_map_->GetId(tag, index);
    auto& packet_idx = packet_index_per_output_[item_id];

    const auto& packets = Outputs().Get(tag, index).packets;
    EXPECT_LT(packet_idx, packets.size());
    const auto& packet = packets.at(packet_idx);
    EXPECT_EQ(expected_timestamp, packet.Timestamp());
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
