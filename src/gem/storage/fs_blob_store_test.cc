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

#include "src/gem/storage/fs_blob_store.h"
#include "src/common/testing/testing.h"

namespace gml {
namespace gem {
namespace storage {

TEST(FilesystemBlobStore, set_and_get) {
  ASSERT_OK_AND_ASSIGN(auto store, FilesystemBlobStore::Create("/tmp/blobs"));
  std::vector<float> floats;
  floats.push_back(1.0);
  floats.push_back(2.0);
  ASSERT_OK(store->Upsert("myfloats", floats.data(), floats.size()));

  ASSERT_OK_AND_ASSIGN(auto received_blob, store->MapReadOnly("myfloats"));

  EXPECT_EQ(2, received_blob->SizeForType<float>());
  EXPECT_EQ(1.0, received_blob->Data<float>()[0]);
  EXPECT_EQ(2.0, received_blob->Data<float>()[1]);
}

}  // namespace storage
}  // namespace gem
}  // namespace gml
