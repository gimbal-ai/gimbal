# Copyright 2023- Gimlet Labs, Inc.
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

load("@rules_oci//oci:defs.bzl", "oci_image")
load("@with_cfg.bzl", "with_cfg")

oci_image_x86_64, _oci_image_x86_64_internal = with_cfg(oci_image).set(
    "platforms",
    [Label("@//bazel/cc_toolchains:linux-x86_64")],
).set(
    Label("@//bazel/cc_toolchains:libc_version"),
    "glibc2_36",
).build()

oci_image_arm64, _oci_image_arm64_internal = with_cfg(oci_image).set(
    "platforms",
    [Label("@//bazel/cc_toolchains:linux-aarch64")],
).set(
    Label("@//bazel/cc_toolchains:libc_version"),
    "glibc2_36",
).build()
