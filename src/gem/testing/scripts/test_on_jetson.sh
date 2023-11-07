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

# Check arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $(basename "$0") <device addr> <bazel target>"
  exit 1
fi

GML_ROOT=$(git rev-parse --show-toplevel)

device_addr="$1"
target="$2"

# Ensure bazel understands the target.
bazel query "${target}"

# Split the argument into two arguments based on the colon delimiter.
IFS=":" read -ra target_parts <<<"$target"

# Check if the input contains a semicolon delimiter
if [ "${#target_parts[@]}" -ne 2 ]; then
  echo "Error: The target should contain exactly one colon delimiter."
  exit 1
fi

# Access the split arguments
bazel_target_path="${target_parts[0]}"
bazel_target="${target_parts[1]}"

# Strip // from the beginning of the value, if present.
bazel_target_path="${bazel_target_path##//}"

bazel run --config=jetson --config=enable_sysroot_debug -c dbg "//${bazel_target_path}:${bazel_target}.tar"
cp "$GML_ROOT/bazel-bin/${bazel_target_path}/${bazel_target}.tar/tarball.tar" "/tmp/${bazel_target}.tar"
chmod 664 "/tmp/${bazel_target}.tar"
scp "/tmp/${bazel_target}.tar" "${device_addr}:/tmp/${bazel_target}.tar"

echo "Now on the device, run:"
echo ""
echo "  docker load -i /tmp/${bazel_target}.tar && docker run --privileged --init --network host --gpus all --runtime nvidia --rm -it --device=/dev/video0:/dev/video0 -v /usr/lib:/host_lib -v /usr/local/cuda-11.4:/host_cuda -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket -v /tmp:/tmp bazel/${bazel_target_path}:${bazel_target}"
echo ""
