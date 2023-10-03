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

load("//bazel/cc_toolchains:utils.bzl", "abi")

_register_toolchain_format = """
def register_toolchain():
    {body}
"""

def only_register_if_not_enabled(rctx, features):
    """Stubs a register_toolchain call if the sysroot features are not enabled.

    Returns:
        boolean, whether the sysroot is enabled or not.
    """
    all_feats_enabled = True
    for feat in features:
        feat_env = "GML_ENABLE_SYSROOT_{feat}".format(feat = feat.upper())
        if feat_env not in rctx.os.environ:
            all_feats_enabled = False

    register = "native.register_toolchains(\"@{name}//:toolchain\")".format(name = rctx.attr.name)
    content = _register_toolchain_format.format(body = register)
    if not all_feats_enabled:
        content = _register_toolchain_format.format(body = "return")

    rctx.file("register_toolchain.bzl", content = content)

    return all_feats_enabled

def _sysroot_repo_impl(rctx):
    if not only_register_if_not_enabled(rctx, rctx.attr.sysroot_features):
        rctx.file("BUILD.bazel")

        # Only download the sysroot if all the feature flags are enabled in the repo_env.
        return

    tar_path = "sysroot.tar.gz"
    rctx.download(
        url = rctx.attr.urls,
        output = tar_path,
        sha256 = rctx.attr.sha256,
    )
    rctx.extract(tar_path)
    extra_target_settings = []
    for feat in rctx.attr.sysroot_features:
        extra_target_settings.append("\"@gml//bazel/cc_toolchains/sysroots:sysroot_{feat}_enabled\"".format(feat = feat))
    for feat in rctx.attr.disabled_for_features:
        extra_target_settings.append("\"@gml//bazel/cc_toolchains/sysroots:sysroot_{feat}_disabled\"".format(feat = feat))
    extra_target_settings = "[" + ", ".join(extra_target_settings) + "]"
    rctx.template(
        "BUILD.bazel",
        Label("@gml//bazel/cc_toolchains/sysroots/{variant}:sysroot.BUILD".format(variant = rctx.attr.variant)),
        substitutions = {
            "{abi}": abi(rctx.attr.target_arch, rctx.attr.libc_version),
            "{extra_target_settings}": extra_target_settings,
            "{libc_version}": rctx.attr.libc_version,
            "{path_to_this_repo}": "external/" + rctx.attr.name,
            "{tar_path}": tar_path,
            "{target_arch}": rctx.attr.target_arch,
        },
    )

sysroot_repo = repository_rule(
    implementation = _sysroot_repo_impl,
    attrs = {
        "disabled_for_features": attr.string_list(default = [], doc = "List of feature flags that should cause this sysroot to be disabled"),
        "libc_version": attr.string(mandatory = True, doc = "Libc version of the sysroot"),
        "sha256": attr.string(mandatory = True, doc = "sha256 of sysroot tarball"),
        "sysroot_features": attr.string_list(default = [], doc = "List of features flags required to enable this sysroot"),
        "target_arch": attr.string(mandatory = True, doc = "CPU Architecture of the sysroot"),
        "urls": attr.string_list(mandatory = True, doc = "list of mirrors to download the sysroot tarball from"),
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

_sysroot_archs = {
    "aarch64": True,
    "x86_64": True,
}
_sysroot_libc_versions = {
    "glibc2_36": True,
}

def _sysroot_repo_name(target_arch, libc_version, variant, features):
    if target_arch not in _sysroot_archs or libc_version not in _sysroot_libc_versions:
        return ""
    name = "sysroot_{target_arch}_{libc_version}_{variant}".format(
        target_arch = target_arch,
        libc_version = libc_version,
        variant = variant,
    )
    if len(features) > 0:
        name = name + "_" + "_".join(features)
    return name

sysroot_repo_name = _sysroot_repo_name
sysroot_libc_versions = list(_sysroot_libc_versions.keys())
sysroot_architectures = list(_sysroot_archs.keys())
