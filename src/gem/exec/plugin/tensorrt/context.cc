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

#include "src/gem/exec/plugin/tensorrt/context.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "src/common/base/base.h"

namespace gml::gem::exec::tensorrt {

cudaStream_t ExecutionContext::CUDAStream() {
  // TODO(james): look into utilizing CUDA streams. For now everything runs fully
  // synchronously on the default stream.
  return nullptr;
}

}  // namespace gml::gem::exec::tensorrt
