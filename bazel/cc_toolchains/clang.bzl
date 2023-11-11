# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
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

load("//bazel:repository_locations.bzl", "REPOSITORY_LOCATIONS")
load("//bazel/cc_toolchains:settings.bzl", "HOST_GLIBC_VERSION")
load("//bazel/cc_toolchains:utils.bzl", "abi")
load("//bazel/cc_toolchains/sysroots:sysroots.bzl", "SYSROOT_ENV_VARS", "only_register_if_not_enabled", "sysroot_repo_name")

def _download_repo(rctx, repo_name, output):
    loc = REPOSITORY_LOCATIONS[repo_name]
    rctx.download_and_extract(
        output = output,
        url = loc["urls"],
        sha256 = loc["sha256"],
        stripPrefix = loc.get("strip_prefix", ""),
    )

def _clang_toolchain_impl(rctx):
    if not only_register_if_not_enabled(rctx, rctx.attr.sysroot_features):
        rctx.file("BUILD.bazel")

        # Only download the toolchain if the sysroot's feature flags are enabled.
        return

    # Unfortunately, we have to download any files that the toolchain uses within this rule.
    toolchain_path = "toolchain"
    _download_repo(rctx, rctx.attr.toolchain_repo, toolchain_path)
    libcxx_path = "libcxx"

    libcxx_repo = "com_llvm_libcxx_{arch}_{libc_version}".format(
        arch = rctx.attr.target_arch,
        libc_version = rctx.attr.libc_version,
    )
    _download_repo(rctx, libcxx_repo, libcxx_path)

    libcxx_build = rctx.read(Label("@gml//bazel/cc_toolchains/clang:libcxx.BUILD"))
    toolchain_files_build = rctx.read(Label("@gml//bazel/cc_toolchains/clang:toolchain_files.BUILD"))

    sysroot_repo = sysroot_repo_name(rctx.attr.target_arch, rctx.attr.libc_version, "build", rctx.attr.sysroot_features)
    sysroot_path = ""
    sysroot_include_prefix = ""
    if sysroot_repo:
        sysroot_path = "external/{repo}".format(repo = sysroot_repo)
        sysroot_include_prefix = "%sysroot%"

    # First combine all of the build file templates into one file.
    rctx.template(
        "BUILD.bazel.tpl",
        Label("@gml//bazel/cc_toolchains/clang:toolchain.BUILD"),
        substitutions = {
            "{libcxx_build}": libcxx_build,
            "{toolchain_files_build}": toolchain_files_build,
        },
    )

    target_libc_constraints = ["@gml//bazel/cc_toolchains:libc_version_{libc_version}".format(libc_version = rctx.attr.libc_version)]

    # Allow host toolchains to be selected regardless of target libc version
    if rctx.attr.use_for_host_tools:
        target_libc_constraints = []

    # Then substitute in parameters into the combined template.
    rctx.template(
        "BUILD.bazel",
        "BUILD.bazel.tpl",
        substitutions = {
            "{clang_major_version}": rctx.attr.clang_version.split(".")[0],
            "{clang_version}": rctx.attr.clang_version,
            "{host_abi}": abi(rctx.attr.host_arch, rctx.attr.host_libc_version),
            "{host_arch}": rctx.attr.host_arch,
            "{libc_version}": rctx.attr.libc_version,
            "{libcxx_path}": libcxx_path,
            "{name}": rctx.attr.name,
            "{sysroot_include_prefix}": sysroot_include_prefix,
            "{sysroot_path}": sysroot_path,
            "{sysroot_repo}": sysroot_repo,
            "{target_abi}": abi(rctx.attr.target_arch, rctx.attr.libc_version),
            "{target_arch}": rctx.attr.target_arch,
            "{target_libc_constraints}": str(target_libc_constraints),
            "{this_repo}": rctx.attr.name,
            "{toolchain_path}": toolchain_path,
            "{use_for_host_tools}": str(rctx.attr.use_for_host_tools),
        },
    )

clang_toolchain = repository_rule(
    _clang_toolchain_impl,
    attrs = dict(
        toolchain_repo = attr.string(mandatory = True),
        target_arch = attr.string(mandatory = True),
        libc_version = attr.string(default = HOST_GLIBC_VERSION),
        host_arch = attr.string(default = "x86_64"),
        host_libc_version = attr.string(default = HOST_GLIBC_VERSION),
        clang_version = attr.string(mandatory = True),
        use_for_host_tools = attr.bool(default = False),
        sysroot_features = attr.string_list(default = []),
    ),
    environ = SYSROOT_ENV_VARS,
)
