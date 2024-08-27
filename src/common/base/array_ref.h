/*
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
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <array>
#include <iterator>
#include <type_traits>

#include <glog/logging.h>

namespace gml {

// NOLINTBEGIN(google-explicit-constructor)

/**
 * ArrayRef is a const reference to contiguous data based on llvm::ArrayRef.
 */
template <typename T>
class ArrayRef {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = const_pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

 public:
  ArrayRef() = default;

  // Construct an ArrayRef from a single element.
  ArrayRef(const T& element) : data_(&element), size_(1) {}

  // Construct an ArrayRef from a pointer and size.
  constexpr ArrayRef(const T* data, size_t size) : data_(data), size_(size) {}

  // Construct an ArrayRef from a continuous range of elements.
  constexpr ArrayRef(const T* begin, const T* end) : data_(begin), size_(end - begin) {}

  // Construct an ArrayRef from a std::vector.
  template <typename A>
  ArrayRef(const std::vector<T, A>& vec) : data_(vec.data()), size_(vec.size()) {}

  // Construct an ArrayRef from a std::array
  template <size_t N>
  constexpr ArrayRef(const std::array<T, N>& arr) : data_(arr.data()), size_(N) {}

  // Construct an ArrayRef from a C array.
  template <size_t N>
  constexpr ArrayRef(const T (&arr)[N]) : data_(arr), size_(N) {}

  // Construct an ArrayRef<const T*> from ArrayRef<T*>. This uses SFINAE to ensure that only
  // ArrayRefs of pointers can be converted.
  template <typename U>
  ArrayRef(const ArrayRef<U*>& arr,
           std::enable_if_t<std::is_convertible_v<U* const*, T const*>>* = nullptr)
      : data_(arr.data()), size_(arr.size()) {}

  // Construct an ArrayRef<const T*> from std::vector<T*>. This uses SFINAE to ensure that only
  // vectors of pointers can be converted.
  template <typename U, typename A>
  ArrayRef(const std::vector<U*, A>& vec,
           std::enable_if_t<std::is_convertible_v<U* const*, T const*>>* = nullptr)
      : data_(vec.data()), size_(vec.size()) {}

  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  bool empty() const { return size_ == 0; }

  const T* data() const { return data_; }

  size_t size() const { return size_; }

  const T& front() const {
    CHECK(!empty());
    return data_[0];
  }

  const T& back() const {
    CHECK(!empty());
    return data_[size_ - 1];
  }

  const T& operator[](size_t idx) const {
    CHECK(idx < size_);
    return data_[idx];
  }

  // Disallow accidental assignment from a temporary.
  //
  // The declaration here is extra complicated so that "arrayRef = {}" continues to select the move
  // assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(U&& Temporary) = delete;

  // Disallow accidental assignment from a temporary.
  //
  // The declaration here is extra complicated so that "arrayRef = {}" continues to select the move
  // assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(data_, data_ + size_); }

 private:
  /// The start of the array, in an external buffer.
  const T* data_ = nullptr;

  /// The number of elements.
  size_type size_ = 0;
};

// NOLINTEND(google-explicit-constructor)
}  // namespace gml
