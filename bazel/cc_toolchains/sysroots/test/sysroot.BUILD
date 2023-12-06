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

load("@gml//bazel/cc_toolchains/sysroots:sysroot_toolchain.bzl", "sysroot_toolchain")

filegroup(
    name = "all_files",
    srcs = glob(
        [
            "**",
        ],
        exclude = ["{tar_path}"],
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "tar",
    srcs = ["{tar_path}"],
    visibility = ["//visibility:public"],
)

sysroot_toolchain(
    name = "sysroot_toolchain",
    architecture = "{target_arch}",
    files = ":all_files",
    path = "{path_to_this_repo}",
    tar = ":tar",
    extra_compile_flags = {extra_compile_flags},
    extra_link_flags = {extra_link_flags},
)

toolchain(
    name = "toolchain",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:{target_arch}",
    ],
    target_settings = [
        "@gml//bazel/cc_toolchains:libc_version_{libc_version}",
    ] + {extra_target_settings},
    toolchain = ":sysroot_toolchain",
    toolchain_type = "@gml//bazel/cc_toolchains/sysroots/test:toolchain_type",
    visibility = ["//visibility:public"],
)
