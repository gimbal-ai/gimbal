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

load("@com_github_fmeum_rules_meta//meta:defs.bzl", "meta")
load("@rules_oci//oci:defs.bzl", "oci_image")

oci_image_x86_64 = meta.wrap_with_transition(
    oci_image,
    {
        "@//bazel/cc_toolchains:libc_version": meta.replace_with("glibc2_36"),
        "//command_line_option:platforms": meta.replace_with("@//bazel/cc_toolchains:linux-x86_64"),
    },
)

oci_image_arm64 = meta.wrap_with_transition(
    oci_image,
    {
        "@//bazel/cc_toolchains:libc_version": meta.replace_with("glibc2_36"),
        "//command_line_option:platforms": meta.replace_with("@//bazel/cc_toolchains:linux-aarch64"),
    },
)
