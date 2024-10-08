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

#include "src/gem/calculators/plugin/cpu_tensor/image_shape_cpu_tensor_calculator.h"

#include <mediapipe/framework/formats/image_frame.h>

#include "src/common/testing/testing.h"
#include "src/gem/exec/plugin/cpu_tensor/context.h"
#include "src/gem/testing/core/calculator_tester.h"
#include "src/gem/testing/plugin/cpu_tensor/operators.h"

namespace gml::gem::calculators::cpu_tensor {

using ::gml::gem::exec::cpu_tensor::CPUTensorPtr;
using ::gml::gem::testing::CalculatorTester;

constexpr char kImageShapeNode[] = R"pbtxt(
calculator: "ImageShapeCPUTensorCalculator"
input_side_packet: "EXEC_CTX:cpu_exec_ctx"
input_stream: "IMAGE_FRAME:image_frame"
output_stream: "IMAGE_SHAPE:image_shape"
)pbtxt";

TEST(ImageShapeCPUTensorCalculator, CorrectImageShape) {
  auto cpu_exec_ctx = std::make_unique<ImageShapeCPUTensorCalculator::ExecutionContext>();

  CalculatorTester tester(kImageShapeNode);

  constexpr int32_t kWidth = 10;
  constexpr int32_t kHeight = 50;

  mediapipe::ImageFrame img_frame(mediapipe::ImageFormat::FORMAT_SRGB, kWidth, kHeight);

  ASSERT_OK_AND_ASSIGN(auto expected_tensor,
                       cpu_exec_ctx->TensorPool()->GetTensor(sizeof(int32_t) * 2));
  auto data = reinterpret_cast<int32_t*>(expected_tensor->data());
  data[0] = kWidth;
  data[1] = kHeight;

  auto ts = mediapipe::Timestamp::Min();
  tester.WithExecutionContext(cpu_exec_ctx.get())
      .ForInput("IMAGE_FRAME", std::move(img_frame), ts)
      .Run()
      .ExpectOutput<CPUTensorPtr>("IMAGE_SHAPE", ts, ::testing::Pointee(*expected_tensor));
}

}  // namespace gml::gem::calculators::cpu_tensor
