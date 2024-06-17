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

#include "src/gem/calculators/core/bytetrack_calculator.h"

#include <random>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/calculator_runner.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/logging.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/testing/core/calculator_tester.h"

namespace gml::gem::calculators::core {

using ::gml::internal::api::core::v1::Detection;

constexpr char kGraph[] = R"pbtxt(
  calculator: "ByteTrackCalculator"
  input_stream: "DETECTIONS:detection_list"
  output_stream: "DETECTIONS:tracked_detection_list"
)pbtxt";

// An absolute noise factor applied to values like x, y, width, and height.
std::uniform_real_distribution<float> kPixelNoiseDist(-0.5, 0.5);

// A multiplicative noise factor.
std::uniform_real_distribution<float> kScoreNoiseDist(0.95, 1.05);

Detection Box1AtTime(int t, std::mt19937* rng) {
  Detection box;

  auto* bbox = box.mutable_bounding_box();
  bbox->set_xc(static_cast<float>((5.0 + (1.0 * t)) + kPixelNoiseDist(*rng)));
  bbox->set_yc(static_cast<float>(75.0 + kPixelNoiseDist(*rng)));
  bbox->set_width(static_cast<float>(10.0 + kPixelNoiseDist(*rng)));
  bbox->set_height(static_cast<float>(10.0 + kPixelNoiseDist(*rng)));

  auto* label = box.add_label();
  label->set_label("person");
  label->set_score(static_cast<float>(0.9 * kScoreNoiseDist(*rng)));

  return box;
}

Detection Box2AtTime(int t, std::mt19937* rng) {
  Detection box;

  auto* bbox = box.mutable_bounding_box();
  bbox->set_xc(static_cast<float>((95.0 - (1.0 * t)) + kPixelNoiseDist(*rng)));
  bbox->set_yc(static_cast<float>(25.0 + kPixelNoiseDist(*rng)));
  bbox->set_width(static_cast<float>(10.0 + kPixelNoiseDist(*rng)));
  bbox->set_height(static_cast<float>(10.0 + kPixelNoiseDist(*rng)));

  auto* label = box.add_label();
  label->set_label("person");
  label->set_score(static_cast<float>(0.7 * kScoreNoiseDist(*rng)));

  return box;
}

TEST(ByteTrackCalculatorTest, TrackTwoSimultaneousObjects) {
  std::mt19937 rng(37);

  mediapipe::CalculatorRunner runner(kGraph);

  // The basic scenario is that there are two boxes:
  // - Box 1: A person starting at approximately (5, 75) with increasing x values.
  // - Box 2: A person starting at approximately (95, 25) with decreasing x values.
  // We expect the tracker will be able to assign tracking IDs to the separate boxes,
  // and to track them through the frames.

  int t;

  t = 1;
  {
    std::vector<Detection> detections;
    detections.push_back(Box1AtTime(t, &rng));
    detections.push_back(Box2AtTime(t, &rng));

    // Run the calculator for a single packet.
    mediapipe::Packet p = mediapipe::MakePacket<std::vector<Detection>>(std::move(detections));
    p = p.At(mediapipe::Timestamp(t));
    runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(p);
  }

  t = 2;
  {
    std::vector<Detection> detections;
    detections.push_back(Box1AtTime(t, &rng));
    detections.push_back(Box2AtTime(t, &rng));

    // Run the calculator for a single packet.
    mediapipe::Packet p = mediapipe::MakePacket<std::vector<Detection>>(std::move(detections));
    p = p.At(mediapipe::Timestamp(t));
    runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(p);
  }

  t = 3;
  {
    std::vector<Detection> detections;
    detections.push_back(Box1AtTime(t, &rng));
    detections.push_back(Box2AtTime(t, &rng));

    // Run the calculator for a single packet.
    mediapipe::Packet p = mediapipe::MakePacket<std::vector<Detection>>(std::move(detections));
    p = p.At(mediapipe::Timestamp(t));
    runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(p);
  }

  LOG(INFO) << "Running graph.";
  ASSERT_OK(runner.Run());

  // Check output.
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(outputs.NumEntries(), 1);

  const std::vector<mediapipe::Packet>& output_packets = outputs.Tag("DETECTIONS").packets;
  EXPECT_EQ(output_packets.size(), 3);

  constexpr double kTolerance = 0.05;

  // TODO(oazizi): Use gtest matchers.

  const auto& detection1 = output_packets[0].Get<std::vector<Detection>>();
  ASSERT_EQ(detection1.size(), 2);
  EXPECT_EQ(detection1[0].track_id().value(), 1);
  EXPECT_EQ(detection1[0].label(0).label(), "id: 1");
  EXPECT_NEAR(detection1[0].label(0).score(), 0.9, kTolerance);
  EXPECT_EQ(detection1[1].track_id().value(), 2);
  EXPECT_EQ(detection1[1].label(0).label(), "id: 2");
  EXPECT_NEAR(detection1[1].label(0).score(), 0.7, kTolerance);

  const auto& detection2 = output_packets[1].Get<std::vector<Detection>>();
  ASSERT_EQ(detection2.size(), 2);
  EXPECT_EQ(detection2[0].track_id().value(), 1);
  EXPECT_EQ(detection2[0].label(0).label(), "id: 1");
  EXPECT_NEAR(detection2[0].label(0).score(), 0.9, kTolerance);
  EXPECT_EQ(detection2[1].track_id().value(), 2);
  EXPECT_EQ(detection2[1].label(0).label(), "id: 2");
  EXPECT_NEAR(detection2[1].label(0).score(), 0.7, kTolerance);

  const auto& detection3 = output_packets[2].Get<std::vector<Detection>>();
  ASSERT_EQ(detection3.size(), 2);
  EXPECT_EQ(detection3[0].track_id().value(), 1);
  EXPECT_EQ(detection3[0].label(0).label(), "id: 1");
  EXPECT_NEAR(detection3[0].label(0).score(), 0.9, kTolerance);
  EXPECT_EQ(detection3[1].track_id().value(), 2);
  EXPECT_EQ(detection3[1].label(0).label(), "id: 2");
  EXPECT_NEAR(detection3[1].label(0).score(), 0.7, kTolerance);
}

}  // namespace gml::gem::calculators::core
