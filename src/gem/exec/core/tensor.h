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

#pragma once

#include <vector>

#include "src/common/base/base.h"
#include "src/gem/exec/core/data_type.h"

namespace gml {
namespace gem {
namespace exec {
namespace core {

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

}  // namespace core
}  // namespace exec
}  // namespace gem
}  // namespace gml
