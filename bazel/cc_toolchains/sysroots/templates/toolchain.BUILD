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

toolchain(
    name = "toolchain",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:{target_arch}",
        "@gml//bazel/cc_toolchains:is_exec_false",
    ],
    target_settings = [
        "@gml//bazel/cc_toolchains:libc_version_{libc_version}",
    ] + {extra_target_settings},
    toolchain = "@{implementation_repo}//:sysroot_toolchain",
    toolchain_type = "@gml//bazel/cc_toolchains/sysroots/{variant}:toolchain_type",
    visibility = ["//visibility:public"],
)
