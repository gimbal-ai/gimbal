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

#include <mediapipe/framework/formats/image_frame.h>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/testing/protobuf.h"
#include "src/common/testing/testing.h"
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/bounding_box_tensors_to_detections_options.pb.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPool;
using ::gml::internal::api::core::v1::Detection;

static constexpr char kBoundingBoxToDetectionsNode[] = R"pbtxt(
calculator: "BoundingBoxTensorsToDetections"
input_stream: "BOX_TENSOR:box_tensor"
input_stream: "SCORE_TENSOR:score_tensor"
input_stream: "INDEX_TENSOR:index_tensor"
input_stream: "ORIG_IMAGE_SHAPE:image_shape"
output_stream: "detection_list"
node_options {
  [type.googleapis.com/gml.gem.calculators.cpu_tensor.optionspb.BoundingBoxToDetectionsOptions] {
    index_to_label: "person"
    index_to_label: "cat"
    index_to_label: "house"
    coordinate_format: $0
  }
}
)pbtxt";

struct BoundingBoxTestCase {
  optionspb::BoundingBoxToDetectionsOptions::CoordinateFormat coordinate_format;
  int width;
  int height;
  std::vector<std::vector<float>> candidates;
  std::vector<std::vector<float>> scores;
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> expected_rects;
  std::vector<std::string> expected_labels;
  std::vector<float> expected_scores;
};

class BoundingBoxTensorsToDetectionsTest : public ::testing::TestWithParam<BoundingBoxTestCase> {};

TEST_P(BoundingBoxTensorsToDetectionsTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  CPUTensorPool pool;
  auto config = absl::Substitute(
      kBoundingBoxToDetectionsNode,
      optionspb::BoundingBoxToDetectionsOptions_CoordinateFormat_Name(test_case.coordinate_format));
  testing::CalculatorTester tester(config);

  int n_candidates = static_cast<int>(test_case.candidates.size());
  int n_selections = static_cast<int>(test_case.indices.size());
  int n_classes = static_cast<int>(test_case.scores.size());

  ASSERT_OK_AND_ASSIGN(auto boxes_tensor,
                       pool.GetTensor(TensorShape{1, n_candidates, 4}, DataType::FLOAT32));

  auto* box_data = boxes_tensor->TypedData<DataType::FLOAT32>();
  for (const auto& candidate : test_case.candidates) {
    CHECK_EQ(4, candidate.size());
    for (auto val : candidate) {
      *box_data = val;
      box_data++;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto score_tensor,
                       pool.GetTensor(TensorShape{1, n_classes, n_candidates}, DataType::FLOAT32));

  auto* score_data = score_tensor->TypedData<DataType::FLOAT32>();
  for (const auto& box_scores : test_case.scores) {
    ASSERT_EQ(n_candidates, box_scores.size());
    for (auto val : box_scores) {
      *score_data = val;
      score_data++;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto index_tensor,
                       pool.GetTensor(TensorShape{n_selections, 3}, DataType::INT32));

  auto* index_data = index_tensor->TypedData<DataType::INT32>();
  for (const auto& index : test_case.indices) {
    ASSERT_EQ(3, index.size());
    for (auto val : index) {
      *index_data = val;
      index_data++;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto orig_image_shape, pool.GetTensor(TensorShape{1, 2}, DataType::FLOAT32));

  auto* shape_data = orig_image_shape->TypedData<DataType::FLOAT32>();
  shape_data[0] = static_cast<float>(test_case.height);
  shape_data[1] = static_cast<float>(test_case.width);

  std::vector<Detection> expected_detections(test_case.expected_rects.size());

  for (size_t i = 0; i < test_case.expected_rects.size(); ++i) {
    auto* d = &expected_detections[i];
    auto* rect = d->mutable_bounding_box();
    rect->set_xc(test_case.expected_rects[i][0]);
    rect->set_yc(test_case.expected_rects[i][1]);
    rect->set_width(test_case.expected_rects[i][2]);
    rect->set_height(test_case.expected_rects[i][3]);

    auto* label = d->add_label();
    label->set_label(test_case.expected_labels[i]);
    label->set_score(test_case.expected_scores[i]);
  }

  tester.ForInput("BOX_TENSOR", std::move(boxes_tensor), 0)
      .ForInput("SCORE_TENSOR", std::move(score_tensor), 0)
      .ForInput("INDEX_TENSOR", std::move(index_tensor), 0)
      .ForInput("ORIG_IMAGE_SHAPE", std::move(orig_image_shape), 0)
      .Run()
      .ExpectOutput<std::vector<Detection>>(
          "", 0, 0,
          ::testing::Pointwise(::gml::testing::proto::ApproxEqProto(), expected_detections));
}

INSTANTIATE_TEST_SUITE_P(
    BoundingBoxToDetectionsSuite, BoundingBoxTensorsToDetectionsTest,
    ::testing::Values(
        BoundingBoxTestCase{
            .coordinate_format =
                optionspb::BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_DIAG_CORNERS_YX,
            .width = 100,
            .height = 400,
            .candidates =
                {
                    {10, 10, 20, 20},
                    {0, 50, 100, 100},
                    {50, 50, 250, 100},
                    {100, 0, 120, 10},
                    {0, 0, 400, 50},
                },
            .scores =
                {
                    {0.1, 0.1, 0.1, 0.5, 0.1},
                    {0.9, 0.1, 0.0, 0.2, 0.1},
                    {0.0, 0.0, 0.01, 0.8, 0.1},
                },
            .indices =
                {
                    {0, 0, 3},
                    {0, 1, 0},
                    {0, 2, 3},
                },
            .expected_rects =
                {
                    {0.05, 0.275, 0.1, 0.05},
                    {0.15, 0.0375, 0.1, 0.025},
                    {0.05, 0.275, 0.1, 0.05},
                },
            .expected_labels =
                {
                    "person",
                    "cat",
                    "house",
                },
            .expected_scores =
                {
                    0.5,
                    0.9,
                    0.8,
                },

        },
        BoundingBoxTestCase{
            .coordinate_format =
                optionspb::BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_CENTER_ANCHORED,
            .width = 100,
            .height = 400,
            .candidates =
                {
                    {10, 10, 20, 20},
                    {0, 50, 100, 100},
                    {50, 50, 50, 100},
                    {80, 0, 10, 10},
                    {0, 0, 10, 50},
                },
            .scores =
                {
                    {0.1, 0.1, 0.1, 0.5, 0.1},
                    {0.9, 0.1, 0.0, 0.2, 0.1},
                    {0.0, 0.0, 0.8, 0.01, 0.1},
                },
            .indices =
                {
                    {0, 0, 3},
                    {0, 1, 0},
                    {0, 2, 2},
                },
            .expected_rects =
                {
                    {0.8, 0.0, 0.1, 0.025},
                    {0.1, 0.025, 0.2, 0.05},
                    {0.5, 0.125, 0.5, 0.25},
                },
            .expected_labels =
                {
                    "person",
                    "cat",
                    "house",
                },
            .expected_scores =
                {
                    0.5,
                    0.9,
                    0.8,
                },

        },
        BoundingBoxTestCase{
            .coordinate_format =
                optionspb::BoundingBoxToDetectionsOptions::COORDINATE_FORMAT_CENTER_ANCHORED,
            .width = 100,
            .height = 400,
            .candidates =
                {
                    {10, 10, 20, 20},
                    {0, 50, 100, 100},
                    {50, 50, 50, 100},
                    {80, 0, 10, 10},
                    {0, 0, 10, 50},
                },
            .scores =
                {
                    {0.1, 0.1, 0.1, 0.5, 0.1},
                    {0.9, 0.1, 0.0, 0.2, 0.1},
                    {0.0, 0.0, 0.8, 0.01, 0.1},
                },
            .indices =
                {
                    {0, 0, 3},
                    {0, 1, 0},
                    {0, -1, 0},
                },
            .expected_rects =
                {
                    {0.8, 0.0, 0.1, 0.025},
                    {0.1, 0.025, 0.2, 0.05},
                },
            .expected_labels =
                {
                    "person",
                    "cat",
                },
            .expected_scores =
                {
                    0.5,
                    0.9,
                },

        },
        BoundingBoxTestCase{
            .coordinate_format = optionspb::BoundingBoxToDetectionsOptions::
                COORDINATE_FORMAT_NORMALIZED_DIAG_CORNERS_XY,
            .width = 100,
            .height = 400,
            .candidates =
                {
                    {0.1, 0.025, 0.2, 0.05},
                    {0, 0.125, 1, 0.25},
                    {0.5, 0.125, 0.6, 0.25},
                    {0.5, 0, 0.6, 0.025},
                    {0, 0, 0.1, 0.125},
                },
            .scores =
                {
                    {0.1, 0.1, 0.1, 0.5, 0.1},
                    {0.9, 0.1, 0.0, 0.2, 0.1},
                    {0.0, 0.0, 0.8, 0.01, 0.1},
                },
            .indices =
                {
                    {0, 0, 3},
                    {0, 1, 0},
                    {0, -1, 0},
                },
            .expected_rects =
                {
                    {0.55, 0.0125, 0.1, 0.025},
                    {0.15, 0.0375, 0.1, 0.025},
                },
            .expected_labels =
                {
                    "person",
                    "cat",
                },
            .expected_scores =
                {
                    0.5,
                    0.9,
                },

        }));

}  // namespace gml::gem::calculators::cpu_tensor
