# Copyright 2023- Gimlet Labs, Inc.
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

load("@gml//bazel/cc_toolchains/sysroots:create_sysroot.bzl", "create_sysroot")
load("@gml//bazel/cc_toolchains/sysroots:sysroot_path.bzl", "sysroot_path_provider")
load("@gml//bazel/cc_toolchains/sysroots:sysroot_toolchain.bzl", "sysroot_toolchain")

create_sysroot(
    name = "sysroot",
    srcs = {sysroot_srcs},
    path_prefix_filters = {path_prefix_filters},
    visibility = ["//visibility:public"],
)

sysroot_toolchain(
    name = "sysroot_toolchain",
    architecture = "{target_arch}",
    extra_compile_flags = {extra_compile_flags},
    extra_link_flags = {extra_link_flags},
    files = ":sysroot_all_files",
    path_info = ":sysroot_all_files",
    tar = ":sysroot",
)

sysroot_path_provider(
    name = "sysroot_path_provider",
    path_info = ":sysroot_all_files",
    visibility = ["//visibility:public"],
)

alias(
    name = "all_files",
    actual = ":sysroot_all_files",
    visibility = ["//visibility:public"],
)
