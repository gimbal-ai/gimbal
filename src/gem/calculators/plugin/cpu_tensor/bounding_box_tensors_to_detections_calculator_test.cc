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
input_stream: "CLASS_TENSOR:class_tensor"
output_stream: "detection_list"
node_options {
  [type.googleapis.com/gml.gem.calculators.cpu_tensor.optionspb.BoundingBoxToDetectionsOptions] {
    index_to_label: "person"
    index_to_label: "cat"
    index_to_label: "house"
  }
}
)pbtxt";

struct BoundingBoxTestCase {
  std::vector<std::vector<float>> selected_boxes;
  std::vector<float> selected_scores;
  std::vector<int> selected_classes;
  std::vector<std::vector<float>> expected_rects;
  std::vector<std::string> expected_labels;
  std::vector<float> expected_scores;
};

class BoundingBoxTensorsToDetectionsTest : public ::testing::TestWithParam<BoundingBoxTestCase> {};

TEST_P(BoundingBoxTensorsToDetectionsTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  CPUTensorPool pool;
  testing::CalculatorTester tester(kBoundingBoxToDetectionsNode);

  int n_selected = static_cast<int>(test_case.selected_boxes.size());

  ASSERT_OK_AND_ASSIGN(auto boxes_tensor,
                       pool.GetTensor(TensorShape{1, n_selected, 4}, DataType::FLOAT32));

  auto* box_data = boxes_tensor->TypedData<DataType::FLOAT32>();
  for (const auto& box : test_case.selected_boxes) {
    CHECK_EQ(4, box.size());
    for (auto val : box) {
      *box_data = val;
      box_data++;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto score_tensor,
                       pool.GetTensor(TensorShape{1, n_selected}, DataType::FLOAT32));

  auto* score_data = score_tensor->TypedData<DataType::FLOAT32>();
  for (auto val : test_case.selected_scores) {
    *score_data = val;
    score_data++;
  }

  ASSERT_OK_AND_ASSIGN(auto class_tensor,
                       pool.GetTensor(TensorShape{1, n_selected}, DataType::INT32));

  auto* class_data = class_tensor->TypedData<DataType::INT32>();
  for (auto class_index : test_case.selected_classes) {
    *class_data = class_index;
    class_data++;
  }

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

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("BOX_TENSOR", std::move(boxes_tensor), ts)
      .ForInput("SCORE_TENSOR", std::move(score_tensor), ts)
      .ForInput("CLASS_TENSOR", std::move(class_tensor), ts)
      .Run()
      .ExpectOutput<std::vector<Detection>>(
          "", 0, ts,
          ::testing::Pointwise(::gml::testing::proto::ApproxEqProto(), expected_detections));
}

INSTANTIATE_TEST_SUITE_P(BoundingBoxToDetectionsSuite, BoundingBoxTensorsToDetectionsTest,
                         ::testing::Values(BoundingBoxTestCase{
                             .selected_boxes =
                                 {
                                     {0.05, 0.275, 0.1, 0.05},
                                     {0.15, 0.0375, 0.1, 0.025},
                                     {0.05, 0.275, 0.1, 0.05},
                                 },
                             .selected_scores =
                                 {
                                     0.1,
                                     0.8,
                                     0.9,
                                 },
                             .selected_classes = {0, 1, 2},
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
                                     0.1,
                                     0.8,
                                     0.9,
                                 },

                         }));

}  // namespace gml::gem::calculators::cpu_tensor
