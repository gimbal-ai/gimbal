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

toolchain(
    name = "toolchain",
    exec_compatible_with = [
        "@platforms//cpu:{host_arch}",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:{target_arch}",
        "@platforms//os:linux",
    ] + (["@gml//bazel/cc_toolchains:is_exec_true"] if {use_for_host_tools} else ["@gml//bazel/cc_toolchains:is_exec_false"]),
    target_settings = [
        "@gml//bazel/cc_toolchains:compiler_clang",
    ] + {extra_target_settings},
    toolchain = "@{impl_repo}//:cc_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
