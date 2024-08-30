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

#include "src/common/safetensors_wrapper/safetensors.h"

#include <exception>
#include <memory>
#include <string>

#include "src/common/base/error.h"
#include "src/common/base/macros.h"
#include "src/common/system/memory_mapped_file.h"

namespace gml::safetensors {

TensorView::TensorView(rust::TensorView rust_view)
    : rust_view_(std::move(rust_view)),
      shape_(rust_view_.shape.begin(), rust_view_.shape.end()),
      data_(reinterpret_cast<const char*>(rust_view_.data.data()), rust_view_.data.size()) {}

rust::Dtype TensorView::DataType() const { return rust_view_.dtype; }

const std::vector<size_t>& TensorView::Shape() const { return shape_; }

std::string_view TensorView::Data() const { return data_; }

size_t TensorView::Offset() const { return rust_view_.offset; }

SafeTensorsFile::SafeTensorsFile(::rust::Box<rust::SafeTensors> rust_safetensors,
                                 std::unique_ptr<const system::MemoryMappedFile> mmap,
                                 PrivateConstructorTag)
    : rust_safetensors_(std::move(rust_safetensors)), mmap_(std::move(mmap)) {
  for (const auto& name : rust::names(*rust_safetensors_)) {
    names_.push_back(static_cast<std::string>(name));
  }
}

StatusOr<std::unique_ptr<SafeTensorsFile>> SafeTensorsFile::Open(
    const std::filesystem::path& path) {
  GML_ASSIGN_OR_RETURN(auto mmap, system::MemoryMappedFile::MapReadOnly(path));
  ::rust::Slice<const uint8_t> buffer(mmap->data(), mmap->size());
  try {
    auto rust_safetensors = rust::deserialize(buffer);
    return std::make_unique<SafeTensorsFile>(std::move(rust_safetensors), std::move(mmap),
                                             PrivateConstructorTag{});
  } catch (const std::exception& exc) {
    return gml::error::Internal("failed to deserialize safetensors file: $0", exc.what());
  }
}

const std::vector<std::string>& SafeTensorsFile::TensorNames() const { return names_; }

size_t SafeTensorsFile::Length() const { return rust::len(*rust_safetensors_); }

size_t SafeTensorsFile::Size() const { return rust::size(*rust_safetensors_); }

StatusOr<std::unique_ptr<TensorView>> SafeTensorsFile::Tensor(const std::string& name) const {
  try {
    auto view = rust::tensor(*rust_safetensors_, ::rust::Str(name));
    return std::make_unique<TensorView>(view);
  } catch (const std::exception&) {
    return gml::error::NotFound("no tensor with name $0", name);
  }
}

}  // namespace gml::safetensors
