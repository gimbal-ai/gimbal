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
