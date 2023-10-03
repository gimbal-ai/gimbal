#!/bin/bash

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

set -e

output_dir="$(realpath "$1")"
docker_image_tag="$2"

# For each arch and variant a sysroot with only default features is produced.
# To enable sysroots with extra features see extra_sysroots below.
architectures=("amd64" "arm64")
variants=("runtime" "build" "test")

extra_sysroots=(
  # Extra sysroots to produce in 'arch variant feat1 feat2' format.
  "amd64 test debug"
  "arm64 test debug"
)

pkgdb_dir="$(mktemp -d)"

download_package_index() {
  arch="$1"
  curl -fL "http://ftp.debian.org/debian/dists/bookworm/main/binary-${arch}/Packages.xz" |
    xz --decompress >"${pkgdb_dir}/${arch}"
}

build_sysroot() {
  arch="$1"
  variant="$2"
  features=("${@:3}")
  sysroot_name="sysroot-${arch}-${variant}"
  for feat in "${features[@]}"; do
    sysroot_name="${sysroot_name}-${feat}"
  done
  echo "Building ${output_dir}/${sysroot_name}.tar.gz"
  docker run -it -v "${output_dir}":/build \
    -v "${pkgdb_dir}":/pkgdb \
    "${docker_image_tag}" \
    "/pkgdb/${arch}" \
    "/build/${sysroot_name}.tar.gz" \
    "${variant}" \
    "${features[@]}"
}

for arch in "${architectures[@]}"; do
  download_package_index "${arch}"
  for variant in "${variants[@]}"; do
    build_sysroot "${arch}" "${variant}"
  done
done

for config in "${extra_sysroots[@]}"; do
  read -ra config_arr <<<"$config"
  arch="${config_arr[0]}"
  variant="${config_arr[1]}"
  features=("${config_arr[@]:2}")
  build_sysroot "${arch}" "${variant}" "${features[@]}"
done

rm -rf "${pkgdb_dir:?}"
