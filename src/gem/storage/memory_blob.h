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

#include "src/common/base/base.h"

namespace gml {
namespace gem {
namespace storage {

/**
 * MemoryBlob is the base class for a binary blob of data in memory.
 * Subclasses are responsible for actually storing the data and implementing data() and size().
 */
class MemoryBlob : gml::NotCopyable {
 public:
  template <typename T>
  T* Data() {
    return reinterpret_cast<T*>(data());
  }

  template <typename T>
  const T* Data() const {
    return reinterpret_cast<const T*>(data());
  }

  template <typename T>
  size_t SizeForType() const {
    static_assert(sizeof(T) >= sizeof(char));
    constexpr size_t data_size = sizeof(T) / sizeof(char);
    return size() / data_size;
  }

  virtual ~MemoryBlob() {}

 protected:
  virtual char* data() = 0;
  virtual const char* data() const = 0;
  virtual size_t size() const = 0;
};

}  // namespace storage
}  // namespace gem
}  // namespace gml
