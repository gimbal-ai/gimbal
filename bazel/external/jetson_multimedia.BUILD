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

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "libargus_headers",
    hdrs = glob(["usr/src/jetson_multimedia_api/include/**/*.h"]),
    includes = ["usr/src/jetson_multimedia_api/include"],
    target_compatible_with = ["@platforms//cpu:aarch64"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "app_framework_api",
    srcs = glob([
      "usr/src/jetson_multimedia_api/samples/common/classes/*.cpp"
    ],
    exclude = [
      "usr/src/jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp",
      "usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp",
      "usr/src/jetson_multimedia_api/samples/common/classes/NvVulkanRenderer.cpp",
      "usr/src/jetson_multimedia_api/samples/common/classes/NvDrmRenderer.cpp",
    ]),
    local_defines=["TEGRA_ACCELERATE"],
    target_compatible_with = ["@platforms//cpu:aarch64"],
    visibility = ["//visibility:public"],
    deps=[":libargus_headers"]
)

