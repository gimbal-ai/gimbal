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

def _collect_shared_libs(srcs, libs):
    for lib in libs:
        native.filegroup(
            name = lib.partition(".")[0],
            srcs = [srcs],
            output_group = lib,
            visibility = ["//visibility:public"],
        )

    native.filegroup(
        name = "shared_libs",
        srcs = [":" + lib.partition(".")[0] for lib in libs],
        visibility = ["//visibility:public"],
    )

collect_shared_libs = _collect_shared_libs
