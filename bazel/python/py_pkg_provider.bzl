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
