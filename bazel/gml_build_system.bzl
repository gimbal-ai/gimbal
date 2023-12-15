# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Based on envoy(28d5f41) envoy/bazel/envoy_build_system.bzl
# Compute the final copts based on various options.

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_test")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//bazel:lib.bzl", "default_arg")
load("//bazel/clang_tidy:clang_tidy.bzl", "clang_tidy")

def gml_copts():
    """Common options for gimlet build

    Returns:
        c-options used in the build.
    """
    posix_options = [
        # Warnings setup.
        "-Wall",
        "-Werror",
        "-Wextra",
        "-Wimplicit-fallthrough",
        "-Wfloat-conversion",
        "-Wno-deprecated-declarations",
    ]

    # Since abseil's BUILD.bazel doesn't provide any system 'includes', add them in manually here.
    # In contrast, libraries like googletest do provide includes, so no need to add those.
    manual_system_includes = [
        "-isystemexternal/abseil-cpp~20230802.0",
        "-isystemexternal/org_tensorflow",
        "-isystemexternal/com_github_google_mediapipe",
    ]

    tcmalloc_flags = select({
        "@gml//bazel:disable_tcmalloc": ["-DABSL_MALLOC_HOOK_MMAP_DISABLE=1"],
        "//conditions:default": ["-DTCMALLOC=1"],
    }) + select({
        "@gml//bazel:debug_tcmalloc": ["-DGML_MEMORY_DEBUG_ENABLED=1"],
        "//conditions:default": [],
    })

    # Leaving this here as an example of how to add compiler dependent_flags.
    compiler_dependent_flags = select({
        "@gml//bazel:gcc_build": [
            # Since we globally disable these warnings in the .bazelrc file,
            # we force them enabled for our own source code.
            "-Werror=stringop-truncation",
            "-Werror=maybe-uninitialized",
        ],
        "//conditions:default": [
        ],
    })

    return posix_options + manual_system_includes + tcmalloc_flags + compiler_dependent_flags

# Compute the final linkopts based on various options.
def gml_linkopts():
    return gml_common_linkopts()

# Compute the test linkopts.
def gml_test_linkopts():
    return gml_common_linkopts()

def gml_common_linkopts():
    return select({
        "//bazel:gcc_build": [
            "-pthread",
            "-llzma",
            "-lrt",
            "-ldl",
            "-Wl,--hash-style=gnu",
            "-lunwind",
        ],
        # The OSX system library transitively links common libraries (e.g., pthread).
        "@bazel_tools//tools/osx:darwin": [],
        "//conditions:default": [
            "-pthread",
            "-lunwind",
            "-llzma",
            "-lrt",
            "-ldl",
            "-Wl,--hash-style=gnu",
        ],
    }) + select({
        "//bazel:use_libcpp": [],
        "//conditions:default": ["-lstdc++fs"],
    })

def gml_defines():
    return ["MAGIC_ENUM_RANGE_MIN=-128", "MAGIC_ENUM_RANGE_MAX=256"]

def _default_external_deps():
    return [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/base:base",
        "@com_google_absl//absl/strings:strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/debugging:failure_signal_handler",
        "@com_google_absl//absl/functional:bind_front",
        "@com_github_neargye_magic_enum//:magic_enum",
    ]

def _default_internal_deps():
    return [
        "//src/common/base:cc_library",
        #        "//src/common/memory:cc_library",
        #        "//src/common/perf:cc_library",
    ]

def gml_default_features():
    return [
        "-external_dep",
    ]

# PL C++ library targets should be specified with this function.
def gml_cc_library_internal(name, **kwargs):
    copts = kwargs.pop("copts", [])
    deps = kwargs.pop("deps", [])
    tcmalloc_dep = kwargs.pop("tcmalloc_dep", False)
    repository = kwargs.pop("repository", "")
    alwayslink = kwargs.pop("alwayslink", True)
    linkstatic = kwargs.pop("linkstatic", True)
    defines = kwargs.pop("defines", [])
    features = kwargs.pop("features", [])
    srcs = kwargs.pop("srcs", [])
    hdrs = kwargs.pop("hdrs", [])
    testonly = kwargs.pop("testonly", False)

    if tcmalloc_dep:
        deps += _tcmalloc_external_deps(repository)

    copts = gml_copts() + copts
    deps = deps + _default_external_deps()
    defines = gml_defines() + defines
    features = features + gml_default_features()

    cc_library(
        name = name,
        copts = copts,
        deps = deps,
        srcs = srcs,
        hdrs = hdrs,
        alwayslink = alwayslink,
        linkstatic = linkstatic,
        defines = defines,
        features = features,
        testonly = testonly,
        **kwargs
    )

    clang_tidy(
        name = name + ".clangtidy",
        srcs = srcs + hdrs,
        library = ":" + name,
        tags = ["manual"],
        copts = copts,
        features = features,
        testonly = testonly,
    )

def gml_cc_library(**kwargs):
    kwargs["deps"] = kwargs.get("deps", [])
    kwargs["deps"] = kwargs["deps"] + _default_internal_deps()

    gml_cc_library_internal(**kwargs)

# PL C++ binary targets should be specified with this function.
def _gml_cc_binary(name, **kwargs):
    copts = kwargs.pop("copts", [])
    deps = kwargs.pop("deps", [])
    linkopts = kwargs.pop("linkopts", [])
    linkstatic = kwargs.pop("linkstatic", True)
    repository = kwargs.pop("repository", "")
    malloc = kwargs.pop("malloc", _tcmalloc_external_dep(repository))
    features = kwargs.pop("features", [])
    defines = kwargs.pop("defines", [])

    copts += gml_copts()
    linkopts += gml_linkopts()
    deps += _default_external_deps()
    deps += _default_internal_deps()
    defines += gml_defines()
    features += gml_default_features()

    cc_binary(
        name = name,
        copts = gml_copts() + copts,
        linkopts = linkopts,
        linkstatic = linkstatic,
        malloc = malloc,
        deps = deps,
        defines = defines,
        features = features,
        **kwargs
    )

# PL C++ test targets should be specified with this function.
def _gml_cc_test(
        name,
        srcs = [],
        data = [],
        repository = "",
        deps = [],
        tags = [],
        shard_count = None,
        size = "small",
        timeout = "short",
        args = [],
        defines = [],
        coverage = True,
        local = False,
        flaky = False,
        include_test_runner = True,
        **kwargs):
    test_lib_tags = list(tags)
    if coverage:
        test_lib_tags.append("coverage_test_lib")
    gml_cc_test_library(
        name = name + "_lib",
        srcs = srcs,
        data = data,
        deps = deps,
        repository = repository,
        tags = test_lib_tags,
        defines = defines,
    )
    if include_test_runner:
        data = data + ["//bazel/test_runners:test_runner_dep"]
    cc_test(
        name = name,
        copts = gml_copts(),
        linkopts = gml_test_linkopts(),
        linkstatic = 1,
        malloc = _tcmalloc_external_dep(repository),
        deps = [
            ":" + name + "_lib",
            repository + "//src/common/testing:test_main",
            repository + "//src/shared/version:test_version_linkstamp",
        ] + _default_external_deps(),
        args = args,
        data = data,
        tags = tags + ["coverage_test"],
        shard_count = shard_count,
        size = size,
        timeout = timeout,
        local = local,
        flaky = flaky,
        features = gml_default_features(),
        **kwargs
    )

# PL C++ test related libraries (that want gtest, gmock) should be specified
# with this function.
def gml_cc_test_library(
        name,
        srcs = [],
        hdrs = [],
        data = [],
        deps = [],
        visibility = None,
        # buildifier: disable=unused-variable
        repository = "",
        tags = [],
        defines = []):
    copts = gml_copts()
    features = gml_default_features()
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        copts = copts,
        testonly = 1,
        deps = deps + [
            "@com_google_googletest//:gtest",
            repository + "//src/common/testing:cc_library",
        ] + _default_external_deps(),
        tags = tags,
        defines = gml_defines() + defines,
        visibility = visibility,
        alwayslink = 1,
        linkstatic = 1,
        features = features,
    )

    clang_tidy(
        name = name + ".clangtidy",
        srcs = srcs + hdrs,
        library = ":" + name,
        tags = ["manual"],
        copts = copts,
        features = features,
        testonly = 1,
    )

# PL C++ mock targets should be specified with this function.
def gml_cc_mock(name, **kargs):
    gml_cc_test_library(name = name, **kargs)

# Dependencies on tcmalloc_and_profiler should be wrapped with this function.
def _tcmalloc_external_dep(repository):
    return select({
        repository + "//bazel:disable_tcmalloc": None,
        "//conditions:default": "//bazel/external:gperftools",
    })

# As above, but wrapped in list form for adding to dep lists. This smell seems needed as
# SelectorValue values have to match the attribute type. See
# https://github.com/bazelbuild/bazel/issues/2273.
def _tcmalloc_external_deps(repository):
    return select({
        repository + "//bazel:disable_tcmalloc": [],
        "//conditions:default": ["//bazel/external:gperftools"],
    })

def _add_test_runner(kwargs):
    if "data" not in kwargs:
        kwargs["data"] = []
    kwargs["data"].append("//bazel/test_runners:test_runner_dep")

def _add_no_sysroot(kwargs):
    if "target_compatible_with" not in kwargs:
        kwargs["target_compatible_with"] = []
    kwargs["target_compatible_with"] = kwargs["target_compatible_with"] + _no_sysroot()

def _no_sysroot():
    return select({
        "//bazel/cc_toolchains:libc_version_glibc_host": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def _jetson_sysroot():
    return select({
        "//bazel/cc_toolchains/sysroots:sysroot_type_jetson": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def _intelgpu_sysroot():
    return select({
        "//bazel/cc_toolchains/sysroots:sysroot_type_intelgpu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def gml_go_test(**kwargs):
    default_arg(kwargs, "linkmode", select({
        "@gml//bazel:no_pie": "auto",
        "//conditions:default": "pie",
    }))
    _add_test_runner(kwargs)
    go_test(**kwargs)

def gml_go_binary(**kwargs):
    default_arg(kwargs, "linkmode", "pie")
    go_binary(**kwargs)

def gml_py_test(**kwargs):
    _add_test_runner(kwargs)
    _add_no_sysroot(kwargs)
    py_test(**kwargs)

def gml_sh_test(**kwargs):
    _add_test_runner(kwargs)
    native.sh_test(**kwargs)

gml_cc_binary = _gml_cc_binary
gml_cc_test = _gml_cc_test
no_sysroot = _no_sysroot
jetson_sysroot = _jetson_sysroot
intelgpu_sysroot = _intelgpu_sysroot
