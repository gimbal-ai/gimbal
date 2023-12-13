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
#include "src/common/bazel/runfiles.h"

#include <memory>

#include "src/common/base/base.h"
#include "src/common/fs/fs_wrapper.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace gml::bazel {

namespace {
std::string g_binary_name;
}

using ::bazel::tools::cpp::runfiles::Runfiles;

void SetBazelBinaryName(std::string_view name) { g_binary_name = name; }

// Binaries that support running outside of bazel (i.e. invoked *not* using "bazel run")
// call this method to find files based on the build manifest created with the build target.
// When invoked using "bazel run" the "runfiles" mechanism used here fails, but, the files
// are found organically based on their relative path.
std::filesystem::path RunfilePath(const std::filesystem::path& rel_path) {
  std::string error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(g_binary_name, gflags::StringFromEnv("RUNFILES_MANIFEST_FILE", ""),
                       gflags::StringFromEnv("RUNFILES_DIR", ""), &error));
  if (!error.empty()) {
    if (!::gml::fs::Exists(rel_path)) {
      char const* const errmsg = "Failed to initialize runfiles, cannot find: $0: $1.";
      LOG(FATAL) << absl::Substitute(errmsg, rel_path.string(), error);
    }
    // else: rel_path exists and there is no need to use "bazel runfiles." (see note above).
    return rel_path;
  }

  const std::string path = runfiles->Rlocation(std::filesystem::path("_main") / rel_path);
  return path;
}

}  // namespace gml::bazel
