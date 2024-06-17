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
#include "src/gem/calculators/plugin/cpu_tensor/optionspb/segmentation_masks_to_proto_options.pb.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::core::DataType;
using ::gml::gem::exec::core::TensorShape;
using ::gml::gem::exec::cpu_tensor::CPUTensorPool;
using ::gml::internal::api::core::v1::Segmentation;

static constexpr char kSegmentationMasksToProtoNode[] = R"pbtxt(
calculator: "SegmentationMasksToProtoCalculator"
input_stream: "MASK_TENSOR:mask_tensor"
output_stream: "segmentation"
node_options {
  [type.googleapis.com/gml.gem.calculators.cpu_tensor.optionspb.SegmentationMasksToProtoOptions] {
    index_to_label: "person"
    index_to_label: "cat"
  }
}
)pbtxt";

struct SegmentationTestCase {
  std::vector<std::vector<std::vector<int8_t>>> binary_masks;
  std::vector<std::string> expected_labels;
  std::vector<std::vector<int>> expected_rle;
  int64_t expected_width;
  int64_t expected_height;
};

class SegmentationMasksToProtoCalculatorTest
    : public ::testing::TestWithParam<SegmentationTestCase> {};

TEST_P(SegmentationMasksToProtoCalculatorTest, ConvertsCorrectly) {
  auto test_case = GetParam();

  CPUTensorPool pool;
  testing::CalculatorTester tester(kSegmentationMasksToProtoNode);

  int n_classes = static_cast<int>(test_case.binary_masks.size());
  int height = static_cast<int>(test_case.binary_masks[0].size());
  int width = static_cast<int>(test_case.binary_masks[0][0].size());

  ASSERT_OK_AND_ASSIGN(auto mask_tensor,
                       pool.GetTensor(TensorShape{1, n_classes, height, width}, DataType::INT8));

  auto* mask_data = mask_tensor->TypedData<DataType::INT8>();
  for (const auto& mask : test_case.binary_masks) {
    for (const auto& row : mask) {
      for (auto val : row) {
        *mask_data = static_cast<int8_t>(val);
        mask_data++;
      }
    }
  }

  Segmentation expected_segmentation;
  expected_segmentation.set_width(test_case.expected_width);
  expected_segmentation.set_height(test_case.expected_height);

  for (size_t i = 0; i < test_case.expected_rle.size(); ++i) {
    auto label = test_case.expected_labels[i];
    auto rle = test_case.expected_rle[i];
    auto* mask = expected_segmentation.add_masks();
    mask->set_label(label);
    mask->mutable_run_length_encoding()->Assign(rle.begin(), rle.end());
  }

  auto ts = mediapipe::Timestamp::Min();
  tester.ForInput("MASK_TENSOR", std::move(mask_tensor), ts)
      .Run()
      .ExpectOutput<Segmentation>("", 0, ts,
                                  gml::testing::proto::EqProtoMsg(expected_segmentation));
}

INSTANTIATE_TEST_SUITE_P(SegmentationMasksToProtoSuite, SegmentationMasksToProtoCalculatorTest,
                         ::testing::Values(SegmentationTestCase{.binary_masks =
                                                                    {
                                                                        {
                                                                            {0, 0, 1, 0},
                                                                            {0, 1, 1, 1},
                                                                        },
                                                                        {
                                                                            {1, 1, 0, 1},
                                                                            {1, 0, 0, 0},
                                                                        },
                                                                    },
                                                                .expected_labels =
                                                                    {
                                                                        "person",
                                                                        "cat",
                                                                    },
                                                                .expected_rle =
                                                                    {
                                                                        {2, 1, 2, 3},
                                                                        {0, 2, 1, 2, 3, 0},
                                                                    },
                                                                .expected_width = 4,
                                                                .expected_height = 2

                         }));

}  // namespace gml::gem::calculators::cpu_tensor
