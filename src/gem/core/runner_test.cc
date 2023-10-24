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
#include <mediapipe/framework/calculator_base.h>
#include <mediapipe/framework/calculator_graph.h>
#include <mediapipe/framework/calculator_registry.h>

#include "src/common/testing/testing.h"
#include "src/gem/core/runner.h"
#include "src/gem/core/spec/execution.pb.h"
#include "src/gem/plugins/registry.h"

namespace gml {
namespace gem {
namespace core {

/**
 * OutputTextSidePacketCalculator outputs the string it's given as side packet STR, everytime an
 * input is received on the TICK input.
 */
class OutputTextSidePacketCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->InputSidePackets().Tag("STR").Set<std::string>();
    cc->Inputs().Tag("TICK").Set<int>();
    cc->Outputs().Tag("STR").Set<std::string>();
    return absl::OkStatus();
  }
  absl::Status Open(mediapipe::CalculatorContext*) override { return absl::OkStatus(); }
  absl::Status Close(mediapipe::CalculatorContext*) override { return absl::OkStatus(); }
  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    auto packet =
        mediapipe::MakePacket<std::string>(cc->InputSidePackets().Tag("STR").Get<std::string>())
            .At(cc->InputTimestamp());
    cc->Outputs().Tag("STR").AddPacket(std::move(packet));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OutputTextSidePacketCalculator);

static constexpr char kExecutionSpecPbtxt[] = R"pbtxt(
graph {
  input_side_packet: "string_to_output"
  input_side_packet: "num_ticks"

  node {
    calculator: "CountingSourceCalculator"
    input_side_packet: "MAX_COUNT:num_ticks"
    output_stream: "tick"
  }

  node {
    calculator: "OutputTextSidePacketCalculator"
    input_side_packet: "STR:string_to_output"
    input_stream: "TICK:tick"
    output_stream: "STR:output"
  }

  output_stream: "output"
}
)pbtxt";

TEST(Runner, run_simple_graph_with_side_packet) {
  spec::ExecutionSpec spec;

  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(kExecutionSpecPbtxt, &spec));

  Runner runner(spec);

  std::string test_str("test1234");
  int num_ticks = 10;

  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.emplace("string_to_output", mediapipe::MakePacket<std::string>(test_str));
  side_packets.emplace("num_ticks", mediapipe::MakePacket<int>(num_ticks));

  ASSERT_OK(runner.Init(side_packets));

  int num_output_packets = 0;
  ASSERT_OK(runner.AddOutputStreamCallback<std::string>(
      "output", [&](const std::string& str, const mediapipe::Timestamp&) {
        EXPECT_EQ(test_str, str);
        num_output_packets++;
        return Status::OK();
      }));

  ASSERT_OK(runner.Start());
  ASSERT_OK(runner.Wait());

  EXPECT_EQ(num_ticks, num_output_packets);
}

}  // namespace core
}  // namespace gem
}  // namespace gml
