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
        name = "sysroot_{variant}_tar".format(variant = variant),
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
