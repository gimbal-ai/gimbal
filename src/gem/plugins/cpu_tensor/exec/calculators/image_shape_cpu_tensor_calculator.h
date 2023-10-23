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

#include "src/gem/plugins/cpu_tensor/exec/calculators/base.h"
#include "src/gem/plugins/cpu_tensor/exec/context.h"

namespace gml {
namespace gem {
namespace cputensor {
namespace calculators {

/**
 * ImageShapeCPUTensorCalculator Graph API:
 *
 *  Input Side Packets:
 *    EXEC_CTX cputensor::ExecutionContext*
 *  Inputs:
 *    IMAGE_FRAME mediapipe::ImageFrame
 *  Outputs:
 *    IMAGE_SHAPE CPUTensorPtr containing the input image width and height ([h, w]).
 *
 */
class ImageShapeCPUTensorCalculator : public ExecutionContextBaseCalculator {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  Status OpenImpl(mediapipe::CalculatorContext* cc, cputensor::ExecutionContext* exec_ctx) override;
  Status ProcessImpl(mediapipe::CalculatorContext* cc,
                     cputensor::ExecutionContext* exec_ctx) override;
  Status CloseImpl(mediapipe::CalculatorContext* cc,
                   cputensor::ExecutionContext* exec_ctx) override;
};

}  // namespace calculators
}  // namespace cputensor
}  // namespace gem
}  // namespace gml
