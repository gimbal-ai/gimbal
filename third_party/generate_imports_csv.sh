#!/bin/bash

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

third_party_path="$1"

# Make sure it ends with a /
if [[ ! "${third_party_path}" =~ .*/$ ]]; then
  third_party_path="${third_party_path}/"
fi

repo_path="$(git rev-parse --show-toplevel)"

imports="$(mktemp)"

bazel run //:gazelle -- -repo_root "${repo_path}" -proto_configs "${repo_path}/bazel/rules_proto_config.yaml" -proto_repo_name "" -proto_imports_out "${imports}"

grep ",${third_party_path}" "${imports}" |
  grep -v "depends" |
  awk -F ',' -v OFS=',' '{ gsub("'"${third_party_path}"'","",$3); print }' >"${third_party_path}imports.csv"
