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
    for src in ctx.attr.srcs:
        all_files = src[DefaultInfo].files.to_list()
        all_files.extend(src[DefaultInfo].default_runfiles.files.to_list())
        for f in all_files:
            if "/site-packages/" in f.path:
                _, _, path = f.path.partition("/site-packages/")
                prefix = "site-packages"
            elif "/dist-packages/" in f.path:
                _, _, path = f.path.partition("/dist-packages/")
                prefix = "dist-packages"
            else:
                continue
            path = "/".join([ctx.attr.python_prefix, prefix, path])
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
        python_prefix = attr.string(
            default = "/usr/local/lib/python3.10",
        ),
    ),
)
