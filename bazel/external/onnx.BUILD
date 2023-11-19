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
    name = "source",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "onnx",
    build_args = [
        "--",  # <- Pass remaining options to the native tool.
        "-j`nproc`",
        "-l`nproc`",
    ],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
        "BUILD_ONNX_PYTHON": "OFF",
        "ONNX_BUILD_TESTS": "OFF",
        "ONNX_DISABLE_EXCEPTIONS": "ON",
        "ONNX_GEN_PB_TYPE_STUBS": "OFF",
        "ONNX_USE_PROTOBUF_SHARED_LIBS": "OFF",

        # Use our own protobuf library.
        "Protobuf_USE_STATIC_LIBS": "ON",
        "Protobuf_INCLUDE_DIR": "$$EXT_BUILD_DEPS/include",
        "Protobuf_LIBRARY": "$$EXT_BUILD_DEPS/lib/libprotobuf.a",
        "Protobuf_FOUND": "TRUE",
        "Protobuf_PROTOC_EXECUTABLE": "$$EXT_BUILD_ROOT/$(location @com_google_protobuf//:protoc)",
    },
    visibility = ["//visibility:public"],
    lib_name = "libonnx",
    lib_source = ":source",
    build_data = [
        "@com_google_protobuf//:protoc",
    ],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)
