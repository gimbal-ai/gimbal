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

#pragma once

#include <vector>

#include "src/common/base/base.h"
#include "src/gem/exec/core/data_type.h"

namespace gml::gem::exec::core {

class TensorShape : public std::vector<int32_t> {
 public:
  using std::vector<int32_t>::vector;
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TensorShape& shape) {
    absl::Format(&sink, "[%s]", absl::StrJoin(shape, ", "));
  }
};

class ReshapeableTensor {
 public:
  Status Reshape(const core::TensorShape& shape) {
    for (auto dim : shape) {
      if (dim < 0) {
        return error::Unimplemented("Tensors don't currently support dynamic shapes");
      }
    }
    // TODO(james): check that the shape is valid for the size of the tensor.
    shape_ = shape;
    return Status::OK();
  }
  void SetDataType(core::DataType data_type) { data_type_ = data_type; }

  const core::TensorShape& Shape() const { return shape_; }
  core::DataType DataType() const { return data_type_; }

 private:
  TensorShape shape_;
  core::DataType data_type_;
};

}  // namespace gml::gem::exec::core
