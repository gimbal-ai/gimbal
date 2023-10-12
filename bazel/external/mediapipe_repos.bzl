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

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")

def http_archive(name, **kwargs):
    if name not in native.existing_rules():
        _http_archive(name = name, **kwargs)

def _mediapipe_repos():
    # These repos are from mediapipe's WORKSPACE file.
    http_archive(
        name = "libyuv",
        sha256 = "2c8a4e90db48856c87fd5fe0e237009eff004a95ac84e4d86b0980f108acbdfb",
        urls = ["https://github.com/lemenkov/libyuv/archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz"],
        build_file = "@com_github_google_mediapipe//third_party:libyuv.BUILD",
    )

    http_archive(
        name = "stblib",
        strip_prefix = "stb-b42009b3b9d4ca35bc703f5310eedc74f584be58",
        sha256 = "13a99ad430e930907f5611325ec384168a958bf7610e63e60e2fd8e7b7379610",
        urls = ["https://github.com/nothings/stb/archive/b42009b3b9d4ca35bc703f5310eedc74f584be58.tar.gz"],
        build_file = "@com_github_google_mediapipe//third_party:stblib.BUILD",
        patches = [
            "@com_github_google_mediapipe//third_party:stb_image_impl.diff",
        ],
        patch_args = [
            "-p1",
        ],
    )

    http_archive(
        name = "build_bazel_rules_nodejs",
        sha256 = "94070eff79305be05b7699207fbac5d2608054dd53e6109f7d00d923919ff45a",
        urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.2/rules_nodejs-5.8.2.tar.gz"],
    )
    http_archive(
        name = "rules_proto_grpc",
        sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
        strip_prefix = "rules_proto_grpc-4.2.0",
        urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
    )

mediapipe_repos = _mediapipe_repos
