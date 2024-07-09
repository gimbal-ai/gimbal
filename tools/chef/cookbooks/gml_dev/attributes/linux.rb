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
default["bazelisk"]["sha256"] =
  "ce52caa51ef9e509fb6b7e5ad892e5cf10feb0794b0aed4d2f36adb00a1a2779"

default["buildozer"]["download_path"] =
  "https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildozer-linux-amd64"
default["buildozer"]["sha256"] =
  "8d5c459ab21b411b8be059a8bdf59f0d3eabf9dff943d5eccb80e36e525cc09d"

default["codecov"]["download_path"] =
  "https://uploader.codecov.io/v0.2.3/linux/codecov"
default["codecov"]["sha256"] =
  "648b599397548e4bb92429eec6391374c2cbb0edb835e3b3f03d4281c011f401"

default["containerregistry"]["download_path"] =
  "https://github.com/google/go-containerregistry/releases/download/v0.20.0/go-containerregistry_Linux_arm64.tar.gz"
default["containerregistry"]["sha256"] =
  "2b0fd5a8e63e12b6d16b99e18d537b7afa4987dabdb5756791eadf0838ea5d79"

default["cosign"]["download_path"] =
  "https://github.com/sigstore/cosign/releases/download/v2.2.4/cosign-linux-amd64"
default["cosign"]["sha256"] =
  "97a6a1e15668a75fc4ff7a4dc4cb2f098f929cbea2f12faa9de31db6b42b17d7"

default["golang"]["download_path"] =
  "https://go.dev/dl/go1.22.5.linux-amd64.tar.gz"
default["golang"]["sha256"] =
  "904b924d435eaea086515bc63235b192ea441bd8c9b198c507e85009e6e4c7f0"

default["golangci-lint"]["download_path"] =
  "https://github.com/golangci/golangci-lint/releases/download/v1.56.2/golangci-lint-1.56.2-linux-amd64.tar.gz"
default["golangci-lint"]["sha256"] =
  "e1c313fb5fc85a33890fdee5dbb1777d1f5829c84d655a47a55688f3aad5e501"

default["nodejs"]["download_path"] =
  "https://nodejs.org/dist/v20.9.0/node-v20.9.0-linux-x64.tar.xz"
default["nodejs"]["sha256"] =
  "9033989810bf86220ae46b1381bdcdc6c83a0294869ba2ad39e1061f1e69217a"

default["prototool"]["download_path"] =
  "https://github.com/uber/prototool/releases/download/v1.10.0/prototool-Linux-x86_64"
default["prototool"]["sha256"] =
  "2247ff34ad31fa7d9433b3310879190d1ab63b2ddbd58257d24c267f53ef64e6"

default["regbot"]["download_path"] =
  "https://github.com/regclient/regclient/releases/download/v0.6.1/regbot-linux-amd64"
default["regbot"]["sha256"] =
  "77784f38be732dc4dd311e004a148fcd07b5c3ad60b4a60575614429d2f9d5d8"

default["regctl"]["download_path"] =
  "https://github.com/regclient/regclient/releases/download/v0.6.1/regctl-linux-amd64"
default["regctl"]["sha256"] =
  "e541327d14c8e6d3a2e4b0dfd76046425a1816879d4f5951042791435dec82e3"

default["regsync"]["download_path"] =
  "https://github.com/regclient/regclient/releases/download/v0.6.1/regsync-linux-amd64"
default["regsync"]["sha256"] =
  "c0723d62a2e60bee30c39c2a4633834fcd209eece2992191cf438335ae6bcd18"

default["shellcheck"]["download_path"] =
  "https://github.com/koalaman/shellcheck/releases/download/v0.9.0/shellcheck-v0.9.0.linux.x86_64.tar.xz"
default["shellcheck"]["sha256"] =
  "700324c6dd0ebea0117591c6cc9d7350d9c7c5c287acbad7630fa17b1d4d9e2f"
