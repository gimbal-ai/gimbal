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

filegroup(
    name = "libcxx_compiler_files",
    srcs = glob(["{libcxx_path}/include/**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libcxx_linker_files",
    srcs = glob(["{libcxx_path}/lib/**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libcxx_all_files",
    srcs = [
        ":libcxx_compiler_files",
        ":libcxx_linker_files",
    ],
    visibility = ["//visibility:public"],
)
