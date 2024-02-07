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

load("//bazel/cc_toolchains/sysroots:sysroot_toolchain.bzl", "SysrootPathInfo")

def _sysroot_path_provider_impl(ctx):
    return [
        platform_common.TemplateVariableInfo({
            "SYSROOT": ctx.attr.path_info[SysrootPathInfo].path,
        }),
    ]

sysroot_path_provider = rule(
    implementation = _sysroot_path_provider_impl,
    attrs = {
        "path_info": attr.label(mandatory = True, doc = "Target providing SysrootPathInfo"),
    },
)
