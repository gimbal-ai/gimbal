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

# Resources created by chef on linux are owned by root:root
default["owner"] = "root"
default["group"] = "root"

default["bazelisk"]["download_path"] =
  "https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64"
default["bazel"]["sha256"] =
  "ce52caa51ef9e509fb6b7e5ad892e5cf10feb0794b0aed4d2f36adb00a1a2779"

default["codecov"]["download_path"] =
  "https://uploader.codecov.io/v0.2.3/linux/codecov"
default["codecov"]["sha256"] =
  "648b599397548e4bb92429eec6391374c2cbb0edb835e3b3f03d4281c011f401"

default["golang"]["download_path"] =
  "https://go.dev/dl/go1.21.3.linux-amd64.tar.gz"
default["golang"]["sha256"] =
  "1241381b2843fae5a9707eec1f8fb2ef94d827990582c7c7c32f5bdfbfd420c8"

default["golangci-lint"]["download_path"] =
  "https://github.com/golangci/golangci-lint/releases/download/v1.55.2/golangci-lint-1.55.2-linux-amd64.tar.gz"
default["golangci-lint"]["sha256"] =
  "ca21c961a33be3bc15e4292dc40c98c8dcc5463a7b6768a3afc123761630c09c"

default["nodejs"]["download_path"] =
  "https://nodejs.org/dist/v20.9.0/node-v20.9.0-linux-x64.tar.xz"
default["nodejs"]["sha256"] =
  "9033989810bf86220ae46b1381bdcdc6c83a0294869ba2ad39e1061f1e69217a"

default["prototool"]["download_path"] =
  "https://github.com/uber/prototool/releases/download/v1.10.0/prototool-Linux-x86_64"
default["prototool"]["sha256"] =
  "2247ff34ad31fa7d9433b3310879190d1ab63b2ddbd58257d24c267f53ef64e6"

default["shellcheck"]["download_path"] =
  "https://github.com/koalaman/shellcheck/releases/download/v0.9.0/shellcheck-v0.9.0.linux.x86_64.tar.xz"
default["shellcheck"]["sha256"] =
  "700324c6dd0ebea0117591c6cc9d7350d9c7c5c287acbad7630fa17b1d4d9e2f"
