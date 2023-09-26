#!/bin/bash

# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file was inspired by:
# https://github.com/envoyproxy/envoy/blob/master/bazel/get_workspace_status

# This script will be run bazel when building process starts to
# generate key-value information that represents the status of the
# workspace. The output should be like
#
# KEY1 VALUE1
# KEY2 VALUE2
#
# If the script exits with non-zero code, it's considered as a failure
# and the output will be discarded.

# If this VERSION file exists then it must have been placed here by a
# distribution doing a non-git, source build.
# Distributions would be expected to echo the commit/tag as BUILD_SCM_REVISION.
if [[ -f "VERSION" && -f "GIT_COMMIT" ]]; then
  echo "STABLE_BUILD_SCM_REVISION $(cat GIT_COMMIT)"
  echo "STABLE_BUILD_TAG $(cat VERSION)"
  echo "STABLE_BUILD_NUMBER ${BUILD_NUMBER}"
  echo "STABLE_BUILD_SCM_STATUS Distribution"
  echo "STABLE_BUILT_BY build-bot"

  # Build buddy tags.
  echo "COMMIT_SHA $(cat GIT_COMMIT)"
  echo "VERSION $(cat VERSION)"
  echo "GIT_TREE_STATUS Clean"
  exit 0
fi

# The code below presents an implementation that works for git repository.
if ! git_rev=$(git rev-parse HEAD); then
  exit 1
fi

# Check whether there are any uncommitted changes.
if git diff-index --quiet HEAD --; then
  tree_status="Clean"
else
  tree_status="Modified"
fi

# Use 0.0.0-dev for all dev build versions. We don't have a version number that is
# shared accross all our binaries which makes it hard to semantically version.
echo "STABLE_BUILD_SCM_REVISION ${git_rev}"
echo "STABLE_BUILD_TAG 0.0.0-dev"
echo "STABLE_BUILD_NUMBER 0"
echo "STABLE_BUILD_SCM_STATUS ${tree_status}"
echo "STABLE_BUILT_BY ${USER}"

# Build buddy variables.
echo "COMMIT_SHA ${git_rev}"
echo "VERSION 0.0.0-dev"
echo "GIT_TREE_STATUS ${tree_status}"
