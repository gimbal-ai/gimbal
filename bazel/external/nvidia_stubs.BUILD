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

load("@rules_cc//cc:defs.bzl", "cc_import")

cc_import(
    name = "libcudart",
    shared_library = select({
        "@platforms//cpu:aarch64": "aarch64/libcudart.so.11.0",
        "@platforms//cpu:x86_64": "x86_64/libcudart.so.11.0",
    }),
    visibility = ["//visibility:public"],
)

cc_import(
    name = "libnvinfer",
    shared_library = select({
        "@platforms//cpu:aarch64": "aarch64/libnvinfer.so.8",
        "@platforms//cpu:x86_64": "x86_64/libnvinfer.so.8",
    }),
    visibility = ["//visibility:public"],
)

cc_import(
    name = "libnvinfer_plugin",
    shared_library = select({
        "@platforms//cpu:aarch64": "aarch64/libnvinfer_plugin.so.8",
        "@platforms//cpu:x86_64": "x86_64/libnvinfer_plugin.so.8",
    }),
    visibility = ["//visibility:public"],
)

cc_import(
    name = "libnvonnxparser",
    shared_library = select({
        "@platforms//cpu:aarch64": "aarch64/libnvonnxparser.so.8",
        "@platforms//cpu:x86_64": "x86_64/libnvonnxparser.so.8",
    }),
    visibility = ["//visibility:public"],
)
