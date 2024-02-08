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
    name = "dnnl",
    build_args = [
        "--",  # <- Pass remaining options to the native tool.
        "-j`nproc`",
        "-l`nproc`",
    ],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
        "DNNL_BUILD_EXAMPLES": "OFF",
        "DNNL_BUILD_TESTS": "OFF",
        "DNNL_CPU_RUNTIME": "TBB",
        # ITT gets statically linked and causes a conflict with TBB which also has it. Remove it for now.
        # TODO(oazizi): Fix build so ITT can be enabled.
        "DNNL_ENABLE_ITT_TASKS": "OFF",
        # DNNL_ENABLE_JIT_PROFILING causes a linking issue, so turn it off. See:
        # https://github.com/oneapi-src/oneDNN/blob/e3243ab905f4171c1d7f5f05b2458b843402ea96/cmake/options.cmake#L218
        "DNNL_ENABLE_JIT_PROFILING": "OFF",
        "DNNL_LIBRARY_TYPE": "STATIC",
        "TBBROOT": "$$EXT_BUILD_DEPS",
    },
    lib_source = ":all",
    out_data_dirs = [
        "lib/pkgconfig",
        "lib/cmake",
    ],
    out_static_libs = [
        "libdnnl.a",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_oneapi_oneTBB//:tbb",
    ],
)
