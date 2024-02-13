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

load("@rules_pkg//pkg:providers.bzl", "PackageFilesInfo")

def _py_pkg_provider_impl(ctx):
    dest_to_src = dict()
    lib_prefix = "/".join([ctx.attr.install_prefix, "lib", ctx.attr.python_version])
    bin_prefix = "/".join([ctx.attr.install_prefix, "bin"])
    for src in ctx.attr.srcs:
        all_files = src[DefaultInfo].files.to_list()
        all_files.extend(src[DefaultInfo].default_runfiles.files.to_list())
        for f in all_files:
            if "/site-packages/" in f.path:
                _, _, path = f.path.partition("/site-packages/")
                prefix = "/".join([lib_prefix, "site-packages"])
            elif "/dist-packages/" in f.path:
                _, _, path = f.path.partition("/dist-packages/")
                prefix = "/".join([lib_prefix, "dist-packages"])
            elif "/bin/" in f.path:
                _, _, path = f.path.partition("/bin/")
                prefix = bin_prefix
            elif "/data/" in f.path:
                _, _, path = f.path.partition("/data/")
                prefix = ctx.attr.install_prefix
            else:
                continue
            path = "/".join([prefix, path])
            dest_to_src[path] = f
    return [
        DefaultInfo(
            files = depset(dest_to_src.values()),
        ),
        PackageFilesInfo(
            dest_src_map = dest_to_src,
            attributes = {},
        ),
    ]

py_pkg_provider = rule(
    implementation = _py_pkg_provider_impl,
    attrs = dict(
        srcs = attr.label_list(
            mandatory = True,
            providers = [DefaultInfo],
        ),
        python_version = attr.string(
            default = "python3.11",
        ),
        install_prefix = attr.string(
            default = "/usr/local",
        ),
    ),
)
