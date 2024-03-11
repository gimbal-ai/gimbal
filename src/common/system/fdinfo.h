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
