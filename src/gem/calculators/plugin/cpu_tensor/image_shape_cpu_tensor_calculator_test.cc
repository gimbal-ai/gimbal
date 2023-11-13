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

#include "src/common/testing/testing.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::gem::exec::cpu_tensor::ExecutionContext;
using ::gml::gem::testing::CalculatorTester;

constexpr char kImageShapeNode[] = R"pbtxt(
calculator: "ImageShapeCPUTensorCalculator"
input_side_packet: "EXEC_CTX:cpu_exec_ctx"
input_stream: "IMAGE_FRAME:image_frame"
output_stream: "IMAGE_SHAPE:image_shape"
)pbtxt";

TEST(ImageShapeCPUTensorCalculator, CorrectImageShape) {
  auto cpu_exec_ctx = std::make_unique<ExecutionContext>();

  CalculatorTester tester(kImageShapeNode);

  constexpr int32_t kWidth = 10;
  constexpr int32_t kHeight = 50;

  mediapipe::ImageFrame img_frame(mediapipe::ImageFormat::Format::ImageFormat_Format_SRGB, kWidth,
                                  kHeight);

  ASSERT_OK_AND_ASSIGN(auto expected_tensor,
                       cpu_exec_ctx->TensorPool()->GetTensor(sizeof(float) * 2));
  auto float_data = reinterpret_cast<float*>(expected_tensor->data());
  float_data[0] = static_cast<float>(kHeight);
  float_data[1] = static_cast<float>(kWidth);

  tester.WithExecutionContext(cpu_exec_ctx.get())
      .ForInput("IMAGE_FRAME", std::move(img_frame), 0)
      .Run()
      .ExpectOutput<CPUTensorPtr>("IMAGE_SHAPE", 0, ::testing::Pointee(*expected_tensor));
}

}  // namespace gml::gem::calculators::cpu_tensor
