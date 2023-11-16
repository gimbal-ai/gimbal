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

# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# Note: this determines the order in which the libraries are passed to the
# linker, so if library A depends on library B, library B must come _after_.
# Hence core is at the bottom.
OPENCV_MODULES = [
    "calib3d",
    "features2d",
    "flann",
    "highgui",
    "video",
    "videoio",
    "imgcodecs",
    "imgproc",
    "core",
]

OPENCV_SHARED_LIBS = False

cmake(
    name = "opencv",
    build_args = [
        "--",  # <- Pass remaining options to the native tool.
        "-j`nproc`",
        "-l`nproc`",
    ],
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        # The module list is always sorted alphabetically so that we do not
        # cause a rebuild when changing the link order.
        "BUILD_LIST": ",".join(sorted(OPENCV_MODULES)),
        "BUILD_TESTS": "OFF",
        "BUILD_PERF_TESTS": "OFF",
        "BUILD_EXAMPLES": "OFF",
        "BUILD_SHARED_LIBS": "ON" if OPENCV_SHARED_LIBS else "OFF",
        "WITH_ITT": "OFF",
        "WITH_IPP": "OFF",
        "WITH_JASPER": "OFF",
        "WITH_JPEG": "ON",
        # TODO(james): enable png support if we need it.
        "WITH_PNG": "OFF",
        "WITH_TIFF": "ON",
        "WITH_OPENCL": "OFF",
        "WITH_WEBP": "OFF",
        # Optimization flags
        "CV_ENABLE_INTRINSICS": "ON",
        "WITH_EIGEN": "ON",
        "WITH_PTHREADS": "ON",
        "WITH_PTHREADS_PF": "ON",
        "WITH_FFMPEG": "ON",
        # COPIED FROM MEDIAPIPE:
        # When building tests, by default Bazel builds them in dynamic mode.
        # See https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary.linkstatic
        # For example, when building //mediapipe/calculators/video:opencv_video_encoder_calculator_test,
        # the dependency //mediapipe/framework/formats:image_frame_opencv will
        # be built as a shared library, which depends on a cv::Mat constructor,
        # and expects it to be provided by the main exacutable. The main
        # executable depends on libimage_frame_opencv.so and links in
        # libopencv_core.a, which contains cv::Mat. However, if
        # libopencv_core.a marks its symbols as hidden, then cv::Mat is in
        # opencv_video_encoder_calculator_test but it is not exported, so
        # libimage_frame_opencv.so fails to find it.
        "OPENCV_SKIP_VISIBILITY_HIDDEN": "ON" if not OPENCV_SHARED_LIBS else "OFF",
        "OPENCV_SKIP_PYTHON_LOADER": "ON",
        "BUILD_opencv_python": "OFF",
        "ENABLE_CCACHE": "OFF",

        # Bypass opencv's cmake search mechanisms for deps we build.
        "HAVE_FFMPEG": "TRUE",
        "FFMPEG_LIBRARIES": ";".join([
            "$$EXT_BUILD_DEPS/ffmpeg/lib/{lib}".format(lib=lib)
            for lib in ["libavcodec.so.58", "libavformat.so.58", "libavutil.so.56", "libswresample.so.3", "libswscale.so.5"]
        ]),
        "FFMPEG_LIBRARY_DIRS": "$$EXT_BUILD_DEPS/ffmpeg/lib",
        "FFMPEG_INCLUDE_DIRS": "$$EXT_BUILD_DEPS/ffmpeg/include",
        "FFMPEG_libavcodec_VERSION": "58.91.100",
        "FFMPEG_libavformat_VERSION": "58.45.100",
        "FFMPEG_libavutil_VERSION": "56.51.100",
        "FFMPEG_libswresample_VERSION": "3.7.100",
        "FFMPEG_libswscale_VERSION": "5.7.100",
    },
    includes = ["opencv4"],
    lib_source = ":all",
    out_shared_libs = [
        "libopencv_{module}.so".format(module=module)
        for module in OPENCV_MODULES
    ] if OPENCV_SHARED_LIBS else [],
    out_static_libs = [
        "opencv4/3rdparty/liblibjpeg-turbo.a",
        "opencv4/3rdparty/liblibtiff.a",
        "opencv4/3rdparty/liblibopenjp2.a",
        "opencv4/3rdparty/libIlmImf.a",
    ] + [
        "libopencv_%s.a" % module
        for module in OPENCV_MODULES
    ] if not OPENCV_SHARED_LIBS else None,
    deps = [
        "@com_github_ffmpeg_ffmpeg//:ffmpeg",
    ],
    visibility = ["//visibility:public"],
)
