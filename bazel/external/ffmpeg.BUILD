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

load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

configure_make(
    name = "ffmpeg",
    args = ["-j`nproc`"],
    lib_source = ":all",
    configure_options = [
        "--disable-static",
        "--enable-shared",
        "--enable-cross-compile",
        "--disable-ffmpeg",
        "--disable-ffplay",
        "--disable-ffprobe",
        # TODO(james): need to install yasm/nasm to get this to work.
        # However, x86 isn't our main target arch (for this) yet so not going to deal with it for now.
        "--disable-x86asm",
        # TODO(james): figure out how to pass `strip` binary from the correct toolchain.
        "--disable-stripping",
    ] + select({
        "@platforms//cpu:aarch64": ["--arch=aarch64"],
        "@platforms//cpu:x86_64": ["--arch=x86_64"],
    }),
    out_shared_libs = [
        "libavcodec.so",
        "libavformat.so",
        "libavutil.so",
        "libswscale.so",
        "libswresample.so",
    ],
    visibility = ["//visibility:public"],
)
