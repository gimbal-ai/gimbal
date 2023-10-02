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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//bazel/cc_toolchains:utils.bzl", "abi")

SYSROOT_LOCATIONS = dict(
    sysroot_x86_64_glibc2_36_runtime = dict(
        sha256 = "d4e45226f3c80f7ce220038b73bb28f135e68ad3bfac481ef0f7473929dcd318",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-amd64-runtime.tar.gz",
        ],
    ),
    sysroot_x86_64_glibc2_36_build = dict(
        sha256 = "fc317e6fbdab470b9363e7c8f6ac8da367430aac0b614b92b74895b7e5982d7e",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-amd64-build.tar.gz",
        ],
    ),
    sysroot_x86_64_glibc2_36_test = dict(
        sha256 = "d864ef1dde6ce6db69301b4d92c43b106c953aa3fd40ac35e1aa2cfd0c412a56",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-amd64-test.tar.gz",
        ],
    ),
    sysroot_x86_64_glibc2_36_debug = dict(
        sha256 = "d13a5ba04c9185e9d4bfee43af918ef6d31286b49b5ef35a00303fb1861fc338",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-amd64-debug.tar.gz",
        ],
    ),
    sysroot_aarch64_glibc2_36_runtime = dict(
        sha256 = "db068f88db690f8dfa93804e8229eefbc1c4dcad1701d888c59751b3c3a14fa8",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-arm64-runtime.tar.gz",
        ],
    ),
    sysroot_aarch64_glibc2_36_build = dict(
        sha256 = "28cd5d029c1dc0857fe1b2c2daa55aa4c50074b0473a342378e041219a197165",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-arm64-build.tar.gz",
        ],
    ),
    sysroot_aarch64_glibc2_36_test = dict(
        sha256 = "b9621067e4c701337ae50da8b040205c351e5eb16600eaf7ff5269d3437ca339",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-arm64-test.tar.gz",
        ],
    ),
    sysroot_aarch64_glibc2_36_debug = dict(
        sha256 = "b785fbdc59220fa3260813c7d23b642689696a95f75a4aac509c5ca112ce2292",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231002122909/sysroot-arm64-debug.tar.gz",
        ],
    ),
)

_sysroot_architectures = ["aarch64", "x86_64"]
_sysroot_libc_versions = ["glibc2_36"]
_sysroot_variants = ["runtime", "build", "test", "debug"]

def _sysroot_repo_name(target_arch, libc_version, variant):
    name = "sysroot_{target_arch}_{libc_version}_{variant}".format(
        target_arch = target_arch,
        libc_version = libc_version,
        variant = variant,
    )
    if name in SYSROOT_LOCATIONS:
        return name
    return ""

def _sysroot_setting_name(target_arch, libc_version):
    return "using_sysroot_{target_arch}_{libc_version}".format(
        target_arch = target_arch,
        libc_version = libc_version,
    )

def _sysroot_repo_impl(rctx):
    loc = SYSROOT_LOCATIONS[rctx.attr.name]
    tar_path = "sysroot.tar.gz"
    rctx.download(
        url = loc["urls"],
        output = tar_path,
        sha256 = loc["sha256"],
    )

    rctx.extract(
        tar_path,
        stripPrefix = loc.get("strip_prefix", ""),
    )

    rctx.template(
        "BUILD.bazel",
        Label("@gml//bazel/cc_toolchains/sysroots/{variant}:sysroot.BUILD".format(variant = rctx.attr.variant)),
        substitutions = {
            "{abi}": abi(rctx.attr.target_arch, rctx.attr.libc_version),
            "{libc_version}": rctx.attr.libc_version,
            "{path_to_this_repo}": "external/" + rctx.attr.name,
            "{tar_path}": tar_path,
            "{target_arch}": rctx.attr.target_arch,
        },
    )

_sysroot_repo = repository_rule(
    implementation = _sysroot_repo_impl,
    attrs = {
        "libc_version": attr.string(mandatory = True, doc = "Libc version of the sysroot"),
        "target_arch": attr.string(mandatory = True, doc = "CPU Architecture of the sysroot"),
        "variant": attr.string(mandatory = True, doc = "Use case variant of the sysroot. One of 'runtime', 'build', or 'test'"),
    },
)

SysrootInfo = provider(
    doc = "Information about a sysroot.",
    fields = ["files", "architecture", "path", "tar"],
)

def _sysroot_toolchain_impl(ctx):
    return [
        platform_common.ToolchainInfo(
            sysroot = SysrootInfo(
                files = ctx.attr.files.files,
                architecture = ctx.attr.architecture,
                path = ctx.attr.path,
                tar = ctx.attr.tar.files,
            ),
        ),
    ]

sysroot_toolchain = rule(
    implementation = _sysroot_toolchain_impl,
    attrs = {
        "architecture": attr.string(mandatory = True, doc = "CPU architecture targeted by this sysroot"),
        "files": attr.label(mandatory = True, doc = "All sysroot files"),
        "path": attr.string(mandatory = True, doc = "Path to sysroot relative to execroot"),
        "tar": attr.label(mandatory = True, doc = "Sysroot tar, used to avoid repacking the sysroot as a tar for docker images."),
    },
)

def _gml_sysroot_deps():
    toolchains = []
    for target_arch in _sysroot_architectures:
        for libc_version in _sysroot_libc_versions:
            for variant in _sysroot_variants:
                repo = _sysroot_repo_name(target_arch, libc_version, variant)
                _sysroot_repo(
                    name = repo,
                    target_arch = target_arch,
                    libc_version = libc_version,
                    variant = variant,
                )
                toolchains.append("@{repo}//:toolchain".format(repo = repo))
    native.register_toolchains(*toolchains)

def _gml_sysroot_settings():
    for target_arch in _sysroot_architectures:
        for libc_version in _sysroot_libc_versions:
            selects.config_setting_group(
                name = _sysroot_setting_name(target_arch, libc_version),
                match_all = [
                    "@platforms//cpu:" + target_arch,
                    "//bazel/cc_toolchains:libc_version_" + libc_version,
                ],
                visibility = ["//visibility:public"],
            )

sysroot_repo_name = _sysroot_repo_name
sysroot_libc_versions = _sysroot_libc_versions
sysroot_architectures = _sysroot_architectures
gml_sysroot_settings = _gml_sysroot_settings
gml_sysroot_deps = _gml_sysroot_deps
