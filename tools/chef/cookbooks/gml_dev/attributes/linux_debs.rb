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

if !platform_family?("debian")
  return
end

# TODO(zasgar): Replace with gml versions.
default["clang-linters"]["version"]    = "15.0-20230921235320"
default["clang-linters"]["deb"]        =
  "https://storage.googleapis.com/gimlet-dev-infra-public/clang/#{default['clang-linters']['version']}/clang-linters-#{default['clang-linters']['version']}.deb"
default["clang-linters"]["deb_sha256"] =
  "d1b57b13c22942fc254b0eabcbdb5f2c8f4c1f5102cb4487c267282f10a78c7f"
