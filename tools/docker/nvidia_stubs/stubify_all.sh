#!/bin/bash -e

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

out_path="$1"
docker_arch="$2"

case "${docker_arch}" in
amd64)
  arch="x86_64"
  ;;
arm64)
  arch="aarch64"
  ;;
esac

tmpdir="$(mktemp -d)"
outdir="${tmpdir}/${arch}"
mkdir "${outdir}"

libs=(
  "/usr/local/cuda-11/lib64/libcudart.so.11.0"
  "/usr/lib/${arch}-linux-gnu/libnvinfer.so.8"
  "/usr/lib/${arch}-linux-gnu/libnvinfer_plugin.so.8"
  "/usr/lib/${arch}-linux-gnu/libnvonnxparser.so.8"
)

for l in "${libs[@]}"; do
  CC=gcc "/scripts/stubify.sh" "${l}" "${outdir}/$(basename "$l")"
done

tar -C "${tmpdir}" -cf "${out_path}" .
