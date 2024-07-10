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

#include <absl/container/flat_hash_map.h>

#include "src/common/base/base.h"

namespace gml::system {

class FDInfoExtension {
 public:
  virtual ~FDInfoExtension() = default;
  static StatusOr<std::unique_ptr<FDInfoExtension>> Parse(std::ifstream* fdinfo);
  enum FDInfoExtType {
    FDINFO_TYPE_UNKNOWN = 0,
    FDINFO_TYPE_DRM = 1,
  };
  explicit FDInfoExtension(FDInfoExtType type) : type_(type) {}

  FDInfoExtType Type() const { return type_; }

  virtual std::string ToString() const = 0;

 private:
  FDInfoExtType type_;
};

class DRMFDInfo : public FDInfoExtension {
 public:
  static StatusOr<std::unique_ptr<FDInfoExtension>> Parse(std::ifstream* fdinfo);

  struct Engine {
    uint64_t busy_ns;
    uint64_t capacity = 1;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Engine& engine) {
      absl::Format(&sink, "Engine{busy_ns=%d, capcity=%d}", engine.busy_ns, engine.capacity);
    }
    // TODO(james): parse cycles/maxfreq.
  };

  std::string_view driver() const { return driver_; }
  std::string_view pdev() const { return pdev_; }
  uint64_t client_id() const { return client_id_; }
  const absl::flat_hash_map<std::string, Engine>& engines() const { return engines_; }

  std::string ToString() const override;

 private:
  DRMFDInfo() : FDInfoExtension(FDInfoExtension::FDINFO_TYPE_DRM) {}

  std::string driver_;
  std::string pdev_;
  uint64_t client_id_;

  absl::flat_hash_map<std::string, Engine> engines_;
  // TODO(james): parse memory regions (only useful for very recent kernel versions).
};

class FDInfo {
 public:
  int fd;
  uint64_t pos;
  int flags;
  int mnt_id;
  int inode;

  std::unique_ptr<FDInfoExtension> ext;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const FDInfo& fdinfo) {
    std::string ext_str;
    if (fdinfo.ext) {
      ext_str = fdinfo.ext->ToString();
    }
    absl::Format(&sink, "FDInfo{fd=%d, pos=%d, flags=%d, mnt_id=%d, inode=%d, ext=%s}", fdinfo.fd,
                 fdinfo.pos, fdinfo.flags, fdinfo.mnt_id, fdinfo.inode, ext_str);
  }
};

Status ParseFDInfo(std::ifstream* in, FDInfo* fdinfo);

}  // namespace gml::system
