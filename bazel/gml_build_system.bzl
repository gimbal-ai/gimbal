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
    manual_system_includes = ["-isystem external/com_google_absl", "-isystem external/org_tensorflow"]

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
def gml_cc_library_internal(
        name,
        srcs = [],
        hdrs = [],
        data = [],
        copts = [],
        includes = [],
        visibility = None,
        tcmalloc_dep = False,
        repository = "",
        linkstamp = None,
        linkopts = [],
        local_defines = [],
        defines = [],
        tags = [],
        testonly = 0,
        deps = [],
        strip_include_prefix = None):
    if tcmalloc_dep:
        deps += _tcmalloc_external_deps(repository)
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        copts = gml_copts() + copts,
        includes = includes,
        visibility = visibility,
        tags = tags,
        deps = deps + _default_external_deps(),
        alwayslink = 1,
        linkstatic = 1,
        linkstamp = linkstamp,
        linkopts = linkopts,
        local_defines = local_defines,
        defines = gml_defines() + defines,
        testonly = testonly,
        strip_include_prefix = strip_include_prefix,
        features = gml_default_features(),
    )

def gml_cc_library(**kwargs):
    kwargs["deps"] = kwargs.get("deps", [])
    kwargs["deps"] = kwargs["deps"] + _default_internal_deps()

    gml_cc_library_internal(**kwargs)

# PL C++ binary targets should be specified with this function.
def _gml_cc_binary(
        name,
        srcs = [],
        data = [],
        args = [],
        testonly = 0,
        visibility = None,
        repository = "",
        stamp = 0,
        tags = [],
        deps = [],
        copts = [],
        linkopts = [],
        defines = []):
    if not linkopts:
        linkopts = gml_linkopts()
    deps = deps
    cc_binary(
        name = name,
        srcs = srcs,
        data = data,
        args = args,
        copts = gml_copts() + copts,
        linkopts = linkopts,
        testonly = testonly,
        linkstatic = 1,
        visibility = visibility,
        malloc = _tcmalloc_external_dep(repository),
        stamp = stamp,
        tags = tags,
        deps = deps + _default_external_deps() + _default_internal_deps(),
        defines = gml_defines() + defines,
        features = gml_default_features(),
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
        data = data + ["//bazel/test_runners:test_runner_dep"],
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
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        copts = gml_copts(),
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
        features = gml_default_features(),
    )

# PL C++ mock targets should be specified with this function.
def gml_cc_mock(name, **kargs):
    gml_cc_test_library(name = name, **kargs)

# Dependencies on tcmalloc_and_profiler should be wrapped with this function.
def _tcmalloc_external_dep(repository):
    return select({
        repository + "//bazel:disable_tcmalloc": None,
        "//conditions:default": "//third_party:gperftools",
    })

# As above, but wrapped in list form for adding to dep lists. This smell seems needed as
# SelectorValue values have to match the attribute type. See
# https://github.com/bazelbuild/bazel/issues/2273.
def _tcmalloc_external_deps(repository):
    return select({
        repository + "//bazel:disable_tcmalloc": [],
        "//conditions:default": ["//third_party:gperftools"],
    })

def _add_no_pie(kwargs):
    if "gc_linkopts" not in kwargs:
        kwargs["gc_linkopts"] = []
    kwargs["gc_linkopts"].append("-extldflags")
    kwargs["gc_linkopts"].append("-no-pie")

def _add_test_runner(kwargs):
    if "data" not in kwargs:
        kwargs["data"] = []
    kwargs["data"].append("//bazel/test_runners:test_runner_dep")

def _add_no_sysroot(kwargs):
    if "target_compatible_with" not in kwargs:
        kwargs["target_compatible_with"] = []
    kwargs["target_compatible_with"] = kwargs["target_compatible_with"] + select({
        "//bazel/cc_toolchains:libc_version_glibc_host": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def gml_go_test(**kwargs):
    _add_no_pie(kwargs)
    _add_test_runner(kwargs)
    go_test(**kwargs)

def gml_go_binary(**kwargs):
    _add_no_pie(kwargs)
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
