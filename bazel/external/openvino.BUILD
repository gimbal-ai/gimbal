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
    srcs = glob(
        ["**"],
        # There is a file docs/img/BASIC_IE_API_workflow_С.svg, where what might seem like a `C` is actually
        # a UTF-8 cyrillic character (0xd0a1). This causes problems for our build, so just exclude the file.
        # We use this method, instead of fixing the file using a patch to remove or rename the file,
        # because the patch gets confused with the cyrillic character.
        exclude = ["docs/img/*"],
    ),
    visibility = ["//visibility:public"],
)

cmake(
    name = "openvino",
    build_args = [
        "--",  # <- Pass remaining options to the native tool.
        "-j`nproc`",
        "-l`nproc`",
    ],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",

        # Plugins.
        "ENABLE_INTEL_CPU": "ON",
        "ENABLE_INTEL_GPU": "OFF",
        "ENABLE_INTEL_GNA": "OFF",
        "ENABLE_HETERO": "ON",
        "ENABLE_MULTI": "ON",
        "ENABLE_AUTO": "ON",
        "ENABLE_AUTO_BATCH": "ON",
        "ENABLE_TEMPLATE": "ON",
        "ENABLE_PROXY": "OFF",

        # Front-ends.
        "ENABLE_OV_ONNX_FRONTEND": "OFF",
        "ENABLE_OV_PADDLE_FRONTEND": "OFF",
        "ENABLE_OV_TF_FRONTEND": "OFF",
        "ENABLE_OV_TF_LITE_FRONTEND": "OFF",
        "ENABLE_OV_PYTORCH_FRONTEND": "OFF",
        "ENABLE_OV_IR_FRONTEND": "OFF",
        "ENABLE_IR_V7_READER": "OFF",

        # Auxiliary parts of the build, disable to save compilation time.
        "ENABLE_TESTS": "OFF",
        "ENABLE_SAMPLES": "OFF",
        "ENABLE_DOCS": "OFF",
        "ENABLE_CLANG_FORMAT": "OFF",
        "ENABLE_NCC_STYLE": "OFF",
        "ENABLE_CPPLINT": "OFF",

        # Other options.
        "OPENVINO_EXTRA_MODULES": "",
        "ENABLE_GAPI_PREPROCESSING": "OFF",
        "ENABLE_PYTHON": "OFF",
        "ENABLE_WHEEL": "OFF",
        "ENABLE_STRICT_DEPENDENCIES": "OFF",

        # Dependencies
        "ENABLE_SYSTEM_PUGIXML": "ON",
        "ENABLE_SYSTEM_PROTOBUF": "OFF",
        "ENABLE_SYSTEM_FLATBUFFERS": "OFF",
        "ENABLE_SYSTEM_TBB": "OFF",
        "ENABLE_SYSTEM_OPENCL": "OFF",
        "ENABLE_TBBBIND_2_5": "OFF",
        "CMAKE_PREFIX_PATH": "$$EXT_BUILD_DEPS/libpugixml",
        "MLAS_LIBRARY_DIRS": "$$EXT_BUILD_DEPS/libmlas/lib",
        "MLAS_INCLUDE_DIRS": "$$EXT_BUILD_DEPS/libmlas/include",
        "DNNL_LIBRARY_DIRS": "$$EXT_BUILD_DEPS/dnnl/lib",
        "DNNL_INCLUDE_DIRS": "$$EXT_BUILD_DEPS/dnnl/include",
        "TBBROOT": "$$EXT_BUILD_DEPS",
    },
    visibility = ["//visibility:public"],
    lib_name = "openvino",
    lib_source = ":source",
    targets = [
        "openvino",
        "openvino_c",
    ],
    defines = [
        "OPENVINO_STATIC_LIBRARY",
    ],
    out_lib_dir = "runtime/lib/intel64",
    out_include_dir = "runtime/include",
    out_static_libs = [
        "libopenvino.a",
        "libopenvino_auto_batch_plugin.a",
        "libopenvino_auto_plugin.a",
        "libopenvino_builders.a",
        "libopenvino_c.a",
        "libopenvino_hetero_plugin.a",
        "libopenvino_intel_cpu_plugin.a",
        "libopenvino_itt.a",
        "libopenvino_reference.a",
        "libopenvino_shape_inference.a",
        "libopenvino_snippets.a",
        "libopenvino_template_plugin.a",
        "libopenvino_util.a",
    ],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_github_zeux_pugixml//:pugixml",
        "@com_github_herumi_xbyak//:xbyak",
        "@com_github_oneapi_oneTBB//:tbb",
        "@com_github_openvinotoolkit_mlas//:mlas",
        "@com_github_openvinotoolkit_oneDNN//:dnnl",
    ],
)
