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

# THIS FILE IS GENERATED BY build_all_sysroots

load("@sysroot_aarch64_glibc2_31_build_jetson//:register_toolchain.bzl", _register0 = "register_toolchain")
load("@sysroot_aarch64_glibc2_31_runtime_jetson//:register_toolchain.bzl", _register1 = "register_toolchain")
load("@sysroot_aarch64_glibc2_31_test_jetson//:register_toolchain.bzl", _register2 = "register_toolchain")
load("@sysroot_aarch64_glibc2_36_build//:register_toolchain.bzl", _register3 = "register_toolchain")
load("@sysroot_aarch64_glibc2_36_runtime//:register_toolchain.bzl", _register4 = "register_toolchain")
load("@sysroot_aarch64_glibc2_36_test//:register_toolchain.bzl", _register5 = "register_toolchain")
load("@sysroot_aarch64_glibc2_36_test_debug//:register_toolchain.bzl", _register6 = "register_toolchain")
load("@sysroot_x86_64_glibc2_36_build//:register_toolchain.bzl", _register7 = "register_toolchain")
load("@sysroot_x86_64_glibc2_36_runtime//:register_toolchain.bzl", _register8 = "register_toolchain")
load("@sysroot_x86_64_glibc2_36_test//:register_toolchain.bzl", _register9 = "register_toolchain")
load("@sysroot_x86_64_glibc2_36_test_debug//:register_toolchain.bzl", _register10 = "register_toolchain")

def _register_sysroots():
    _register0()
    _register1()
    _register2()
    _register3()
    _register4()
    _register5()
    _register6()
    _register7()
    _register8()
    _register9()
    _register10()

register_sysroots = _register_sysroots
