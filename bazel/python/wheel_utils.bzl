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

def _abi_tag():
    """Get the ABI tag for a non-pure python wheel."""

    # TODO(james): select based on python version.
    return "cp311"

def _version():
    return select({
        "//bazel:stamped": "{STABLE_BUILD_TAG}",
        "//conditions:default": "0.0.0-dev",
    })

def _platform():
    """Get the platform tag for a non-pure python wheel."""
    return select({
        "//bazel/python:manylinux_2_31_aarch64": "manylinux_2_31_aarch64",
        "//bazel/python:manylinux_2_31_x86_64": "manylinux_2_31_x86_64",
        "//bazel/python:manylinux_2_36_aarch64": "manylinux_2_36_aarch64",
        "//bazel/python:manylinux_2_36_x86_64": "manylinux_2_36_x86_64",
        "//conditions:default": "linux_x86_64",
    })

abi_tag = _abi_tag

# ABI and PY tags are the same for our purposes.
python_tag = _abi_tag
version = _version
platform = _platform
