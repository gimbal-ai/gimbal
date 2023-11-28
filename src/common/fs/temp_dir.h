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

#include <filesystem>
#include <memory>

#include "src/common/base/statusor.h"

namespace gml::fs {

/**
 * Returns a unique temporary directory that is automatically deleted.
 */
class TempDir {
 public:
  static StatusOr<std::unique_ptr<TempDir>> Create();

  std::filesystem::path path() { return dir_; }

  ~TempDir();

 private:
  explicit TempDir(std::filesystem::path dir) : dir_(std::move(dir)) {}

  std::filesystem::path dir_;
};

}  // namespace gml::fs