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
    build_data = [
        "@com_google_protobuf//:protoc",
    ],
    cache_entries = {
        "BUILD_ONNX_PYTHON": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "ONNX_BUILD_TESTS": "OFF",
        "ONNX_DISABLE_EXCEPTIONS": "ON",
        "ONNX_GEN_PB_TYPE_STUBS": "OFF",
        "ONNX_ML": "OFF",
        "ONNX_USE_PROTOBUF_SHARED_LIBS": "OFF",
        "Protobuf_FOUND": "TRUE",
        "Protobuf_INCLUDE_DIR": "$$EXT_BUILD_DEPS/include",
        "Protobuf_LIBRARY": "$$EXT_BUILD_DEPS/lib/libprotobuf.a",
        "Protobuf_PROTOC_EXECUTABLE": "$$EXT_BUILD_ROOT/$(location @com_google_protobuf//:protoc)",

        # Use our own protobuf library.
        "Protobuf_USE_STATIC_LIBS": "ON",
    },
    defines = ["ONNX_NAMESPACE=onnx"],
    lib_source = ":source",
    out_data_dirs = ["lib/cmake"],
    out_static_libs = [
        "libonnx.a",
        "libonnx_proto.a",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)
