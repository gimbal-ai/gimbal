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

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mkdirs", "pkg_mklink", "strip_prefix")

symlinks = {symlinks}

empty_dirs = {empty_dirs}

regular_files = {regular_files}

[
    pkg_mklink(
        name = name,
        link_name = path,
        target = target,
    )
    for name, (path, target) in symlinks.items()
]

pkg_mkdirs(
    name = "dirs",
    dirs = empty_dirs,
)

[
    pkg_files(
        name = "files_w_mode_{}".format(mode),
        srcs = files,
        attributes = pkg_attributes(
            mode = mode,
        ),
        strip_prefix = strip_prefix.from_pkg(),
    )
    for mode, files in regular_files.items()
]

pkg_filegroup(
    name = "all_files",
    srcs = list(symlinks.keys()) + {deps} + (
        [":dirs"] if empty_dirs else []
    ) + [
        ":files_w_mode_{}".format(mode)
        for mode in regular_files
    ],
    visibility = ["//visibility:public"],
)
