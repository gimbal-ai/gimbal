# Copyright 2022 The MediaPipe Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories", "yarn_install")
load("@com_github_google_mediapipe//third_party:external_files.bzl", "external_files")
load("@com_github_google_mediapipe//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

def _mediapipe_deps():
    # These deps are from mediapipe's WORKSPACE file.
    flatbuffers()

    node_repositories()
    yarn_install(
        name = "mediapipe_npm",
        package_json = "@com_github_google_mediapipe//:package.json",
        yarn_lock = "@com_github_google_mediapipe//:yarn.lock",
    )

    rules_proto_grpc_toolchains()
    rules_proto_grpc_repos()

    external_files()

mediapipe_deps = _mediapipe_deps
