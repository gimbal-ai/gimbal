# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

load("@gml//bazel/cc_toolchains/sysroots:create_sysroot.bzl", "create_sysroot")
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
    tar = ":sysroot",
    path_info = ":sysroot_all_files",
)

alias(
    name = "all_files",
    actual = ":sysroot_all_files",
    visibility = ["//visibility:public"],
)
