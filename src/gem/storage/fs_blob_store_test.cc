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

#include "src/gem/storage/fs_blob_store.h"

#include "src/common/system/linux_file_wrapper.h"
#include "src/common/system/memory_mapped_file.h"
#include "src/common/testing/testing.h"

namespace gml::gem::storage {

TEST(FilesystemBlobStore, SetAndGet) {
  ASSERT_OK_AND_ASSIGN(auto store, FilesystemBlobStore::Create("/tmp/blobs"));
  std::vector<float> floats;
  floats.push_back(1.0);
  floats.push_back(2.0);
  ASSERT_OK(store->Upsert("myfloats", floats.data(), floats.size()));

  ASSERT_OK_AND_ASSIGN(auto blob_path, store->FilePath("myfloats"));

  ASSERT_OK_AND_ASSIGN(auto mmap, system::MemoryMappedFile::MapReadOnly(blob_path));

  ASSERT_EQ(2 * sizeof(float), mmap->size());
  EXPECT_EQ(1.0, reinterpret_cast<const float*>(mmap->data())[0]);
  EXPECT_EQ(2.0, reinterpret_cast<const float*>(mmap->data())[1]);
}

}  // namespace gml::gem::storage
