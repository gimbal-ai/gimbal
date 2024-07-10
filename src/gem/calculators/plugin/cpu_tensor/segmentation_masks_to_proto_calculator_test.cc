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
    index_to_label: "dog"
  }
}
)pbtxt";

struct SegmentationTestCase {
  std::vector<std::vector<int32_t>> label_masks;
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

  int height = static_cast<int>(test_case.label_masks.size());
  int width = static_cast<int>(test_case.label_masks[0].size());

  ASSERT_OK_AND_ASSIGN(auto mask_tensor,
                       pool.GetTensor(TensorShape{1, 1, height, width}, DataType::INT32));

  auto* mask_data = mask_tensor->TypedData<DataType::INT32>();
  for (const auto& row : test_case.label_masks) {
    for (auto val : row) {
      *mask_data = static_cast<int32_t>(val);
      mask_data++;
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
                         ::testing::Values(SegmentationTestCase{.label_masks =
                                                                    {
                                                                        {0, 0, 1, 2},
                                                                        {2, 2, 1, 1},
                                                                    },
                                                                .expected_labels =
                                                                    {
                                                                        "person",
                                                                        "cat",
                                                                        "dog",
                                                                    },
                                                                .expected_rle =
                                                                    {
                                                                        {0, 2, 6, 0},
                                                                        {2, 1, 3, 2},
                                                                        {3, 3, 2, 0},
                                                                    },
                                                                .expected_width = 4,
                                                                .expected_height = 2

                         }));

}  // namespace gml::gem::calculators::cpu_tensor
