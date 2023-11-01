#!/bin/bash

# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

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
