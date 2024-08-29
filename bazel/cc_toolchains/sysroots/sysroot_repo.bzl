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

def _sysroot_toolchain_repo_impl(rctx):
    rctx.template(
        "BUILD.bazel",
        Label("@gml//bazel/cc_toolchains/sysroots/templates:toolchain.BUILD"),
        substitutions = {
            "{extra_target_settings}": str(rctx.attr.target_settings),
            "{implementation_repo}": rctx.attr.implementation_repo,
            "{libc_version}": rctx.attr.libc_version,
            "{target_arch}": rctx.attr.target_arch,
            "{variant}": rctx.attr.variant,
        },
    )

_sysroot_toolchain_repo = repository_rule(
    implementation = _sysroot_toolchain_repo_impl,
    attrs = dict(
        libc_version = attr.string(mandatory = True),
        target_arch = attr.string(mandatory = True),
        variant = attr.string(mandatory = True),
        implementation_repo = attr.string(mandatory = True),
        target_settings = attr.string_list(),
    ),
)

def _sanitize_repo_name(name):
    chars = []
    for c in name.elems():
        if c.isalnum() or c == "." or c == "_" or c == "~" or c == "-":
            chars.append(c)
        else:
            chars.append("_")
    return "".join(chars)

def _sysroot_implementation_repo_impl(rctx):
    sysroot_srcs = []
    for pkg in rctx.attr.packages:
        repo_name = "_".join([pkg, rctx.attr.target_arch])
        repo_name = _sanitize_repo_name(repo_name)
        sysroot_srcs.append("@{}//:all_files".format(repo_name))

    sysroot_srcs += rctx.attr.srcs

    rctx.template(
        "BUILD.bazel",
        Label("@gml//bazel/cc_toolchains/sysroots/templates:impl.BUILD"),
        substitutions = {
            "{extra_compile_flags}": str(rctx.attr.extra_compile_flags),
            "{extra_link_flags}": str(rctx.attr.extra_link_flags),
            "{libc_version}": rctx.attr.libc_version,
            "{path_prefix_filters}": str(rctx.attr.path_prefix_filters),
            "{sysroot_srcs}": str(sysroot_srcs),
            "{target_arch}": rctx.attr.target_arch,
        },
    )

_sysroot_implementation_repo = repository_rule(
    implementation = _sysroot_implementation_repo_impl,
    attrs = dict(
        extra_compile_flags = attr.string_list(
            doc = "Extra compile flags used when building with the sysroot",
        ),
        extra_link_flags = attr.string_list(
            doc = "Extra link flags used when building with the sysroot",
        ),
        target_arch = attr.string(
            mandatory = True,
            doc = "Target architecture for sysroot",
        ),
        packages = attr.string_list(
            mandatory = True,
            doc = "A list of debian packages to include in the sysroot, in format '<repo>_<pkg_name>'",
        ),
        path_prefix_filters = attr.string_list(
            doc = "A list of path prefixes to include in the sysroot. If empty list, no filtering is performed. Otherwise only paths with the given prefixes are kept.",
        ),
        srcs = attr.string_list(
            doc = "Optional list of explicitly declared targets the sysroot should include. These targets must provide one of the `rules_pkg` packaging providers.",
        ),
        libc_version = attr.string(
            mandatory = True,
            doc = "Version of libc used by this sysroot",
        ),
    ),
)

def sysroot_repo(
        name,
        libc_version,
        supported_archs,
        variant,
        packages = [],
        srcs = [],
        target_settings = [],
        extra_compile_flags = [],
        extra_link_flags = [],
        path_prefix_filters = []):
    """Creates bazel repos for the given sysroot specification.

    Creates two separate bazel repos, one containing the implementation of the sysroot toolchain,
    the other containing the definition of the sysroot toolchain."""

    # Make sure every sysroot has /tmp.
    srcs = srcs + ["@gml//bazel/cc_toolchains/sysroots:tmp_dir"]

    for target_arch in supported_archs:
        name_w_arch = "_".join([name, target_arch])
        _sysroot_toolchain_repo(
            name = name_w_arch + "_toolchain",
            libc_version = libc_version,
            target_arch = target_arch,
            variant = variant,
            implementation_repo = name_w_arch,
            target_settings = target_settings,
        )

        _sysroot_implementation_repo(
            name = name_w_arch,
            target_arch = target_arch,
            packages = packages,
            srcs = srcs,
            extra_compile_flags = extra_compile_flags,
            extra_link_flags = extra_link_flags,
            path_prefix_filters = path_prefix_filters,
            libc_version = libc_version,
        )

        native.register_toolchains("@{name}_toolchain//:toolchain".format(name = name_w_arch))
