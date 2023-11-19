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

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "tbb",
    build_args = [
        "--",  # <- Pass remaining options to the native tool.
        "-j`nproc`",
        "-l`nproc`",
    ],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
        "TBB_TEST": "OFF",
        "TBB_BUILD": "ON",
        "TBBMALLOC_BUILD": "ON",
        "TBBMALLOC_PROXY_BUILD": "ON",
    },
    visibility = ["//visibility:public"],
    lib_source = ":all",
    out_static_libs = [
        "libtbb.a",
        "libtbbmalloc.a",
    ],
    out_data_dirs = [
        "lib/pkgconfig",
        "lib/cmake",
    ],
)
