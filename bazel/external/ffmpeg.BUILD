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

load("@gml//bazel:foreign_cc.bzl", "collect_shared_libs")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

SHARED_LIBS = [
    "libavdevice.so.60",
    "libavfilter.so.9",
    "libavcodec.so.60",
    "libavformat.so.60",
    "libswresample.so.4",
    "libswscale.so.7",
    "libavutil.so.58",
]

configure_make(
    name = "ffmpeg_lib",
    args = ["-j`nproc`"],
    configure_options = [
        "--disable-static",
        "--enable-shared",
        "--enable-cross-compile",
        "--disable-ffplay",
        "--enable-libopenh264",
        "--ar=$$EXT_BUILD_ROOT/$(AR)",
        "--cc=$$EXT_BUILD_ROOT/$(CC)",
        "--cxx=$$EXT_BUILD_ROOT/$(CC)",
        "--nm=$$EXT_BUILD_ROOT/$(NM)",
        "--strip=$$EXT_BUILD_ROOT/$(STRIP)",
    ] + select({
        "@platforms//cpu:aarch64": ["--arch=aarch64"],
        "@platforms//cpu:x86_64": ["--arch=x86_64"],
    }) + select({
        "@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda": [
            "--enable-cuda",
            "--enable-cuda-llvm",
            "--enable-cuda-nvcc",
            "--enable-cuvid",
            "--enable-ffnvcodec",
            "--enable-libnpp",
            "--enable-libvpx",
            "--enable-nonfree",
            "--enable-nvdec",
            "--nvcc=$$EXT_BUILD_ROOT/$(SYSROOT)/usr/local/cuda/bin/nvcc",
            "--nvccflags=\"-gencode arch=compute_75,code=sm_75\"",
            "--x86asmexe=$$EXT_BUILD_ROOT/$(SYSROOT)/usr/bin/nasm",
        ],
        "//conditions:default": [],
    }),
    env = select({
        "@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda": {
            "PKG_CONFIG_PATH": "$${EXT_BUILD_DEPS}/openh264/lib/pkgconfig:$${EXT_BUILD_DEPS}/ffmpeg_nv_codec_headers/lib/pkgconfig",
        },
        "//conditions:default": {
            "PKG_CONFIG_PATH": "$${EXT_BUILD_DEPS}/openh264/lib/pkgconfig",
        },
    }),
    lib_source = ":all",
    out_binaries = [
        "ffmpeg",
        "ffprobe",
    ],
    out_shared_libs = SHARED_LIBS,
    toolchains = select({
        "@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda": [
            "@sysroot_cuda_build_x86_64//:sysroot_path_provider",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_cisco_openh264//:openh264",
    ] + select({
        "@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda": [
            "@com_github_ffmpeg_nv_codec_headers//:ffmpeg_nv_codec_headers",
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "ffmpeg_bin",
    srcs = [":ffmpeg_lib"],
    output_group = "ffmpeg",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ffprobe_bin",
    srcs = [":ffmpeg_lib"],
    output_group = "ffprobe",
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "ffmpeg_libs",
    srcs = [":shared_libs"],
    package_dir = select({
        "@platforms//cpu:aarch64": "/usr/lib/aarch64-linux-gnu",
        "@platforms//cpu:x86_64": "/usr/lib/x86_64-linux-gnu",
    }),
)

pkg_tar(
    name = "ffmpeg_binaries",
    srcs = [
        ":ffmpeg_bin",
        ":ffprobe_bin",
    ],
    package_dir = "/usr/local/bin",
)

collect_shared_libs(
    srcs = ":ffmpeg_lib",
    libs = SHARED_LIBS,
)
