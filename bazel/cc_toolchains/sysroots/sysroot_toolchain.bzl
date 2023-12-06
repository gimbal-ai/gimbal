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

SysrootInfo = provider(
    doc = "Information about a sysroot.",
    fields = [
        "architecture",
        "extra_compile_flags",
        "extra_link_flags",
        "files",
        "path",
        "tar",
    ],
)

def _sysroot_toolchain_impl(ctx):
    sysroot_path = ctx.attr.path
    extra_compile_flags = [
        flag.replace("%sysroot%", sysroot_path)
        for flag in ctx.attr.extra_compile_flags
    ]
    extra_link_flags = [
        flag.replace("%sysroot%", sysroot_path)
        for flag in ctx.attr.extra_link_flags
    ]
    return [
        platform_common.ToolchainInfo(
            sysroot = SysrootInfo(
                architecture = ctx.attr.architecture,
                extra_compile_flags = extra_compile_flags,
                extra_link_flags = extra_link_flags,
                files = ctx.attr.files.files,
                path = sysroot_path,
                tar = ctx.attr.tar.files,
            ),
        ),
    ]

sysroot_toolchain = rule(
    implementation = _sysroot_toolchain_impl,
    attrs = {
        "architecture": attr.string(mandatory = True, doc = "CPU architecture targeted by this sysroot"),
        "extra_compile_flags": attr.string_list(doc = "Extra compile_flags to use when building with the sysroot. %sysroot% will be expanded to the path of the sysroot."),
        "extra_link_flags": attr.string_list(doc = "Extra link_flags to use when building with the sysroot. %sysroot% will be expanded to the path of the sysroot."),
        "files": attr.label(mandatory = True, doc = "All sysroot files"),
        "path": attr.string(mandatory = True, doc = "Path to sysroot"),
        "tar": attr.label(mandatory = True, doc = "Sysroot tar, used to avoid repacking the sysroot as a tar for docker images."),
    },
)

def _sysroot_build_files_impl(ctx):
    sysroot_toolchain = ctx.toolchains["@gml//bazel/cc_toolchains/sysroots/build:toolchain_type"]
    if not sysroot_toolchain:
        return DefaultInfo(files = depset([]))
    sysroot_info = sysroot_toolchain.sysroot
    return DefaultInfo(files = depset(transitive = [sysroot_info.files]))

sysroot_build_files = rule(
    implementation = _sysroot_build_files_impl,
    toolchains = [
        config_common.toolchain_type("@gml//bazel/cc_toolchains/sysroots/build:toolchain_type", mandatory = False),
    ],
)
