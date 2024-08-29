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

load("@bazel_skylib//lib:paths.bzl", "paths")

def _symlink_tree_impl(ctx):
    files = []
    gen_root = "/".join([ctx.bin_dir.path, ctx.attr.old_root])
    for src in ctx.attr.srcs:
        for file in src[DefaultInfo].files.to_list():
            if file.path.startswith(ctx.attr.old_root):
                rel_path = paths.relativize(file.path, ctx.attr.old_root)
            elif file.path.startswith(gen_root):
                rel_path = paths.relativize(file.path, gen_root)
            else:
                continue

            new_file = ctx.actions.declare_file("/".join([ctx.attr.new_root, rel_path]))
            ctx.actions.symlink(output = new_file, target_file = file)
            files.append(new_file)

    return DefaultInfo(
        files = depset(
            files,
        ),
    )

symlink_tree = rule(
    doc = """
Symlink a tree of files to a new location. The tree structure below `old_root` will be maintained and put at `new_root`.
""",
    implementation = _symlink_tree_impl,
    attrs = dict(
        srcs = attr.label_list(
            mandatory = True,
            doc = "Sources to copy",
        ),
        old_root = attr.string(
            mandatory = True,
            doc = "Path to consider as the root of the old tree",
        ),
        new_root = attr.string(
            default = ".",
            doc = "Relative path below this package to copy the tree into. Defaults to '.'",
        ),
    ),
)
