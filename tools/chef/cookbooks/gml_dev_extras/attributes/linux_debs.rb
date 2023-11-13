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

default["clang"]["version"]    = "15.0-20231108142731"
default["clang"]["deb"]        =
  "https://storage.googleapis.com/gimlet-dev-infra-public/clang/#{default['clang']['version']}/clang-#{default['clang']['version']}.deb"
default["clang"]["deb_sha256"] =
  "6e9f167dff3b5e5dc07361b6ce31b14636b820a4895d31cb5d65efe4a638f097"

# default['gperftools']['version']    = '2.10-pl1'
# default['gperftools']['deb']        =
#   "https://github.com/pixie-io/dev-artifacts/releases/download/gperftools%2F#{default['gperftools']['version']}/gperftools-pixie-#{default['gperftools']['version']}.deb"
# default['gperftools']['deb_sha256'] =
#   '0920a93a8a8716b714b9b316c8d7e8f2ecc242a85147f7bec5e1543d88c203dc'
