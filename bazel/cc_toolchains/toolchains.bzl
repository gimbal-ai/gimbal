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

load("//bazel/cc_toolchains:clang.bzl", "clang_toolchain")

def _gml_cc_toolchains():
    clang_toolchain(
        name = "clang-15.0-x86_64",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "x86_64",
        clang_version = "15.0.6",
        libc_version = "glibc_host",
    )
    clang_toolchain(
        name = "clang-15.0-x86_64-glibc2.36-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "x86_64",
        clang_version = "15.0.6",
        libc_version = "glibc2_36",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:use_debian12_build_sysroot"],
    )
    clang_toolchain(
        name = "clang-15.0-aarch64-glibc2.36-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "aarch64",
        clang_version = "15.0.6",
        libc_version = "glibc2_36",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:use_debian12_build_sysroot"],
    )
    clang_toolchain(
        name = "clang-15.0-aarch64-glibc2.31-jetson-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "aarch64",
        clang_version = "15.0.6",
        libc_version = "glibc2_31",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_jetson"],
    )
    clang_toolchain(
        name = "clang-15.0-x86_64-glibc2.36-non-free-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "x86_64",
        clang_version = "15.0.6",
        libc_version = "glibc2_36",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda"],
    )
    clang_toolchain(
        name = "clang-15.0-exec",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "x86_64",
        clang_version = "15.0.6",
        libc_version = "glibc_host",
        use_for_host_tools = True,
    )
    clang_toolchain(
        name = "clang-15.0-x86_64-glibc2.31-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "x86_64",
        clang_version = "15.0.6",
        libc_version = "glibc2_31",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_debian11"],
        disable_libcxx = True,
        gcc_version = "10",
    )
    clang_toolchain(
        name = "clang-15.0-aarch64-glibc2.31-sysroot",
        toolchain_repo = "com_llvm_clang_15",
        target_arch = "aarch64",
        clang_version = "15.0.6",
        libc_version = "glibc2_31",
        use_sysroot = True,
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_debian11"],
        disable_libcxx = True,
        gcc_version = "10",
    )

gml_cc_toolchains = _gml_cc_toolchains
