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

#include <gtest/gtest.h>
#include <mediapipe/framework/subgraph.h>

#include "src/common/testing/testing.h"

using mediapipe::SubgraphRegistry;

TEST(Subgraph, Basic) {
  // These are subgraphs that were included as the test's dependencies in the BUILD.bazel file.
  // Some of these have node options too, to check that functionality.
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("VideoEncoderSubgraph"));
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("OpenCVCamSourceSubgraph"));

  // These subgraphs were not included in the test's dependencies.
  EXPECT_FALSE(SubgraphRegistry::IsRegistered("YoloModelTensorRTSubgraph"));
}
