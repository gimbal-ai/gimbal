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

def _tar_from_sysroot_info(ctx, sysroot_info):
    out = ctx.actions.declare_file(ctx.attr.name + ".tar")
    ctx.actions.symlink(output = out, target_file = sysroot_info.tar.to_list()[0])
    return [
        DefaultInfo(
            files = depset([out]),
        ),
    ]

def _sysroot_variant_tar_factory(variant):
    def _impl(ctx):
        sysroot_toolchain = ctx.toolchains["//bazel/cc_toolchains/sysroots/{variant}:toolchain_type".format(variant = variant)]
        if sysroot_toolchain == None:
            return [DefaultInfo(files = ctx.attr._empty_tar.files)]
        return _tar_from_sysroot_info(ctx, sysroot_toolchain.sysroot)

    return rule(
        attrs = dict(
            _empty_tar = attr.label(
                default = "//bazel/cc_toolchains/sysroots:empty_tar",
                allow_single_file = True,
            ),
        ),
        implementation = _impl,
        toolchains = [
            config_common.toolchain_type("//bazel/cc_toolchains/sysroots/{variant}:toolchain_type".format(variant = variant), mandatory = False),
        ],
    )

sysroot_runtime_tar = _sysroot_variant_tar_factory("runtime")
sysroot_test_tar = _sysroot_variant_tar_factory("test")
