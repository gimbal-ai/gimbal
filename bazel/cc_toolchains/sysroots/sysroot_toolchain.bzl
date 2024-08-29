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

SysrootInfo = provider(
    doc = "Information about a sysroot.",
    fields = [
        "architecture",
        "extra_compile_flags",
        "extra_link_flags",
        "files",
        "path",
        "tar",
        "libc_version",
    ],
)

SysrootPathInfo = provider(
    doc = "Path to sysroot",
    fields = ["path"],
)

def _sysroot_toolchain_impl(ctx):
    sysroot_path = ctx.attr.path_info[SysrootPathInfo].path
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
                libc_version = ctx.attr.libc_version,
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
        "libc_version": attr.string(mandatory = True, doc = "Version of glibc used by the toolchain"),
        "path_info": attr.label(mandatory = True, doc = "Target providing SysrootPathInfo"),
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
