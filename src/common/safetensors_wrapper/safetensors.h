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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/common/base/statusor.h"
#include "src/common/safetensors_wrapper/src/lib.rs.h"
#include "src/common/system/memory_mapped_file.h"

namespace gml::safetensors {

class TensorView {
 public:
  explicit TensorView(rust::TensorView rust_view);
  rust::Dtype DataType() const;
  const std::vector<size_t>& Shape() const;
  std::string_view Data() const;

 private:
  rust::TensorView rust_view_;
  std::vector<size_t> shape_;
  std::string_view data_;
};

class SafeTensorsFile {
 private:
  struct PrivateConstructorTag {};

 public:
  SafeTensorsFile() = delete;
  SafeTensorsFile(::rust::Box<rust::SafeTensors> rust_safetensors,
                  std::unique_ptr<const system::MemoryMappedFile> mmap, PrivateConstructorTag);

  static StatusOr<std::unique_ptr<SafeTensorsFile>> Open(const std::filesystem::path& path);

  const std::vector<std::string>& TensorNames() const;
  size_t Size() const;
  StatusOr<std::unique_ptr<TensorView>> Tensor(const std::string& name) const;

 private:
  ::rust::Box<rust::SafeTensors> rust_safetensors_;
  std::vector<std::string> names_;
  std::unique_ptr<const system::MemoryMappedFile> mmap_;
};

}  // namespace gml::safetensors
