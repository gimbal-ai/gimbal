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

load("@rules_oci//oci:defs.bzl", "oci_image")
load("@with_cfg.bzl", "with_cfg")

oci_image_x86_64, _oci_image_x86_64_internal = with_cfg(oci_image).set(
    "platforms",
    [Label("@//bazel/cc_toolchains:linux-x86_64")],
).set(
    Label("@//bazel/cc_toolchains:libc_version"),
    "glibc2_36",
).build()

oci_image_arm64, _oci_image_arm64_internal = with_cfg(oci_image).set(
    "platforms",
    [Label("@//bazel/cc_toolchains:linux-aarch64")],
).set(
    Label("@//bazel/cc_toolchains:libc_version"),
    "glibc2_36",
).build()
