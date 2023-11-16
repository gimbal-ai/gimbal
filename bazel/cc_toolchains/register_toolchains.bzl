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

load("@clang-15.0-aarch64-glibc2.31-jetson-sysroot//:register_toolchain.bzl", _register_jetson_sysroot = "register_toolchain")
load("@clang-15.0-aarch64-glibc2.36-sysroot//:register_toolchain.bzl", _register_aarch64_sysroot = "register_toolchain")
load("@clang-15.0-exec//:register_toolchain.bzl", _register_exec = "register_toolchain")
load("@clang-15.0-x86_64//:register_toolchain.bzl", _register = "register_toolchain")
load("@clang-15.0-x86_64-glibc2.36-sysroot//:register_toolchain.bzl", _register_x86_sysroot = "register_toolchain")

def _gml_register_cc_toolchains():
    _register()
    _register_x86_sysroot()
    _register_aarch64_sysroot()
    _register_jetson_sysroot()
    _register_exec()

    native.register_toolchains(
        "//bazel/cc_toolchains:cc-toolchain-gcc-x86_64-gnu",
    )

gml_register_cc_toolchains = _gml_register_cc_toolchains
