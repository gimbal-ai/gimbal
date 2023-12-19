/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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
#include <string>

namespace gml::bazel {

/**
 * Set the binary name. Used for determining the bazel runfiles path
 */
void SetBazelBinaryName(int argc, char** argv);

/**
 * Returns the path to a runfile, specified by a path relative to ToT.
 * Path is valid when run through bazel.
 */
std::filesystem::path RunfilePath(const std::filesystem::path& rel_path);

}  // namespace gml::bazel
