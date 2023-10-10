#!/bin/bash -e

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
