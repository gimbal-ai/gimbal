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

load("@gml//bazel/cc_toolchains/sysroots:sysroot_toolchain.bzl", "sysroot_build_files")
load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@unix_cc_toolchain_config//:cc_toolchain_config.bzl", "cc_toolchain_config")

# buildifier: disable=no-effect
{libcxx_build}

# buildifier: disable=no-effect
{toolchain_files_build}

toolchain_identifier = "{name}_toolchain"

tool_paths = {
    "ar": "{toolchain_path}/bin/llvm-ar",
    "clang-tidy": "{toolchain_path}/bin/clang-tidy",
    "cpp": "{toolchain_path}/bin/clang-cpp",
    "dwp": "{toolchain_path}/bin/llvm-dwp",
    "gcc": "{toolchain_path}/bin/clang-15",
    "ld": "{toolchain_path}/bin/ld.lld",
    "llvm-cov": "{toolchain_path}/bin/llvm-cov",
    "nm": "{toolchain_path}/bin/llvm-nm",
    "objcopy": "{toolchain_path}/bin/llvm-objcopy",
    "objdump": "{toolchain_path}/bin/llvm-objdump",
    "strip": "{toolchain_path}/bin/llvm-strip",
}

includes = [
    "{toolchain_path}/lib/clang/15.0.6/include",
    "{sysroot_include_prefix}/usr/local/include",
    "{sysroot_include_prefix}/usr/include/x86_64-linux-gnu",
    "{sysroot_include_prefix}/usr/include",
    "{sysroot_include_prefix}/usr/include/c++/{gcc_version}",
    "{sysroot_include_prefix}/usr/include/x86_64-linux-gnu/c++/{gcc_version}",
    "{sysroot_include_prefix}/usr/include/c++/{gcc_version}/backward",
] + (["{libcxx_path}/include/c++/v1"] if "{libcxx_path}" else [])

cc_toolchain_config(
    name = "toolchain_config",
    abi_libc_version = "{libc_version}",
    abi_version = "clang",
    compile_flags = [
        "-target",
        "{target_arch}-linux-gnu",
        "-fstack-protector",
        "-Wall",
        "-Wthread-safety",
        "-Wself-assign",
        "-Wunused-but-set-parameter",
        "-fcolor-diagnostics",
        "-fno-omit-frame-pointer",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "{target_arch}",
    cxx_builtin_include_directories = includes,
    cxx_flags = [
        "-std=c++17",
        "-fPIC",
    ],
    dbg_compile_flags = ["-g"],
    enable_sanitizers = not {use_for_host_tools},
    host_system_name = "{host_arch}-unknown-linux-{host_abi}",
    libclang_rt_path = "external/{this_repo}/{toolchain_path}/lib/clang/{clang_version}/lib/linux",
    libcxx_path = "external/{this_repo}/{libcxx_path}" if "{libcxx_path}" else "",
    link_flags = [
        "-target",
        "{target_arch}-linux-gnu",
        "-fuse-ld=lld",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-Bexternal/{this_repo}/{toolchain_path}/bin",
        "-lm",
        "-lgcc_s",
    ] + (["-no-pie"] if {use_for_host_tools} else []),
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "{libc_version}",
    target_system_name = "{target_arch}-unknown-linux-{target_abi}",
    tool_paths = tool_paths,
    toolchain_identifier = toolchain_identifier,
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ],
)

sysroot_build_files(name = "sysroot_files")

filegroup(
    name = "all_files",
    srcs = [
        ":sysroot_files",
        ":toolchain_all_files",
    ] + ([":libcxx_all_files"] if "{libcxx_path}" else []),
)

filegroup(
    name = "ar_files",
    srcs = [
        ":toolchain_ar_files",
    ],
)

filegroup(
    name = "as_files",
    srcs = [
        ":toolchain_as_files",
    ],
)

filegroup(
    name = "compiler_files",
    srcs = [
        ":sysroot_files",
        ":toolchain_compiler_files",
    ] + ([":libcxx_compiler_files"] if "{libcxx_path}" else []),
)

filegroup(
    name = "dwp_files",
    srcs = [
        ":toolchain_dwp_files",
    ],
)

filegroup(
    name = "linker_files",
    srcs = [
        ":sysroot_files",
        ":toolchain_linker_files",
    ] + ([":libcxx_linker_files"] if "{libcxx_path}" else []),
)

filegroup(
    name = "objcopy_files",
    srcs = [
        ":toolchain_objcopy_files",
    ],
)

filegroup(
    name = "strip_files",
    srcs = [
        ":toolchain_strip_files",
    ],
)

cc_toolchain(
    name = "cc_toolchain",
    all_files = ":all_files",
    ar_files = ":ar_files",
    as_files = ":as_files",
    compiler_files = ":compiler_files",
    dwp_files = ":dwp_files",
    exec_transition_for_inputs = False,
    linker_files = ":linker_files",
    module_map = None,
    objcopy_files = ":objcopy_files",
    strip_files = ":strip_files",
    supports_param_files = 1,
    toolchain_config = "toolchain_config",
    toolchain_identifier = toolchain_identifier,
    visibility = ["//visibility:public"],
)
