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

load("@gml//bazel:gml_build_system.bzl", "cuda_sysroot")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

make(
    name = "ffmpeg_nv_codec_headers",
    args = ["-j`nproc`"],
    lib_source = ":all",
    out_headers_only = True,
    out_data_dirs = [
        "lib/pkgconfig",
    ],
    visibility = ["//visibility:public"],
    target_compatible_with = cuda_sysroot(),
)
