#!/usr/bin/env bash

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

################################################################
# Gimlet Installer
################################################################
set -u
set -e

# Trap ctrl-c and call ctrl_c() to reset terminal.
trap ctrl_c INT

function ctrl_c() {
  stty sane
  exit
}

# String formatting functions.
if [[ -t 1 ]]; then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi

tty_reset="$(tty_escape "0")"
tty_bold="$(tty_escape "1")"
tty_underline="$(tty_escape "4")"

tty_red="$(tty_escape "31")"
tty_green="$(tty_escape "32")"
tty_orange="$(tty_escape "33")"
tty_cyan="$(tty_escape "36")"

function emph() {
  printf "${tty_bold}${tty_cyan}%s${tty_reset}\n" "$1"
}

function warn() {
  printf "${tty_bold}${tty_orange}%s${tty_reset}\n" "$1"
}

function fatal() {
  printf "${tty_bold}${tty_red}%s${tty_reset}\n" "$1"
  exit 1
}

function device_type() {
  if [[ "$(uname -m)" == "x86_64" ]]; then
    if [[ -f "/etc/OpenCL/vendors/intel.icd" ]] && [[ -d "/dev/dri" ]]; then
      echo "x86_64 Intel GPU"
      return
    fi
    echo "x86_64 Generic"
    return
  elif [[ "$(uname -m)" == "aarch64" ]]; then
    if [[ "$(uname -r)" != *"-tegra" ]]; then
      echo "aarch64 Generic"
      return
    fi

    if [[ ! -f /proc/device-tree/model ]]; then
      echo "aarch64 Generic"
      return
    fi

    read -r MODEL </proc/device-tree/model
    if [[ "$MODEL" == "NVIDIA Orin Nano"* ]]; then
      echo "aarch64 NVIDIA Orin Nano"
      return
    fi
  fi
  echo "Unknown"
  return
}

# Do not inline this variable; it is the keyword of templating process.
DEFAULT_CONTROLPLANE_ADDR="app.dev.gimletlabs.dev:443"

# Externally settable environment variables.
CONTROLPLANE_ADDR=${GML_CONTROLPLANE_ADDR:-${DEFAULT_CONTROLPLANE_ADDR}}
DEPLOY_KEY=${GML_DEPLOY_KEY:-""}
IMAGE_REPO=${GML_IMAGE_REPO:-"us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/gem_image"}
IMAGE_VERSION=${GML_IMAGE_VERSION:-"latest"}
IMAGE_TAG=${GML_IMAGE_TAG:-""}
IMAGE_DIGEST=${GML_IMAGE_DIGEST:-""}
HOST_NETWORK=${GML_HOST_NETWORK:-"false"}
VIDEO_SOURCE=${GML_VIDEO_SOURCE:-""}
RANDOMIZE_DEVICE_SERIAL=${GML_RANDOMIZE_DEVICE_SERIAL:-"false"}
DEV_MODE=${GML_DEV_MODE:-"false"}
CACHE_DIR=${GML_CACHE_DIR:-"$HOME/.cache/gml"}

mkdir -p "$CACHE_DIR"

docker_flags=(
  -h "$(hostname)"
  --pid=host
  -v "$CACHE_DIR:/gml"
  -v /usr/lib:/host_lib
  # Mount /sys so that GEM can use the mac address as the SERIAL_NUMBER and also read system metrics.
  -v /sys:/host/sys
  # Resolve the image tag on every deploy. This ensures that latest doesn't point to an older release.
  --pull always
)

cmdline_opts=(
  "--blob_store_dir" "/gml"
  "--sys_path" "/host/sys"
  "--sys_class_net_path" "/host/sys/class/net"
)

if [[ "$DEV_MODE" == "true" ]]; then
  warn "DEV MODE: ENABLED"
  docker_flags+=(
    # Run the container interactively so that users can see the logs
    -it
  )
else
  # In production mode, we want to run the container in a detached state.
  docker_flags+=(-d)
fi

if [[ "$HOST_NETWORK" == "true" ]]; then
  docker_flags+=(
    --network=host
  )
fi

if [[ "$VIDEO_SOURCE" == "file://"* ]]; then
  echo "Using file as video source"
  video_file_path=${VIDEO_SOURCE#file://}
  video_filename=$(basename "$video_file_path")
  docker_flags+=(
    -v "$video_file_path:/gml_videos/${video_filename}"
  )
  cmdline_opts+=("--video_source=/gml_videos/${video_filename}")
elif [[ "$VIDEO_SOURCE" == "rtsp://"* ]] || [[ "$VIDEO_SOURCE" == "rtsps://"* ]]; then
  cmdline_opts+=("--video_source=${VIDEO_SOURCE}")
elif [[ -n "$VIDEO_SOURCE" ]]; then
  warn "Video source must be a file path beginning with file:// or an RTSP URL beginning with rtsp:// or rtsps://."
  exit 1
fi

if [[ "$RANDOMIZE_DEVICE_SERIAL" == "true" ]]; then
  cmdline_opts+=("--device_serial=$(</dev/urandom tr -dc 'a-f0-9' | fold -w 32 | head -n 1)")
fi

function add_device_flags() {
  if [[ -n "$VIDEO_SOURCE" ]]; then
    return
  fi
  ret=0
  devs=$(ls /dev/video* 2>/dev/null) || ret=$?
  if [[ $ret -ne 0 ]]; then
    warn "Could not find any video devices in /dev. Trying again with sudo."
    # Perhaps the user isn't in the video group, try sudo.
    devs=$(sudo ls /dev/video* 2>/dev/null)
  fi

  for vid in $devs; do
    docker_flags+=("--device" "${vid}")
  done
}

device_str=$(device_type)

IMAGE_TYPE=""
if [[ "$device_str" == "aarch64 NVIDIA Orin Nano"* ]]; then
  IMAGE_TYPE=jetson
  docker_flags+=(
    --privileged
    --runtime nvidia
    --gpus all
    -v /tmp/argus_socket:/tmp/argus_socket
    -v /usr/local/cuda:/host_cuda
    -e LD_LIBRARY_PATH=/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/host_lib/aarch64-linux-gnu:/host_cuda/lib64:/host_lib/aarch64-linux-gnu/tegra:/host_lib/aarch64-linux-gnu/tegra-egl
  )
elif [[ "$device_str" == "x86_64 Intel GPU" ]]; then
  IMAGE_TYPE=intelgpu
  add_device_flags
  docker_flags+=(
    "--device" "/dev/dri"
    # We need to add /usr/local/lib to the library search path.
    "--env" "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib"
    # We need privileged to collect system GPU metrics.
    "--privileged"
  )
elif [[ "$device_str" == "x86_64"* ]]; then
  IMAGE_TYPE=""
  add_device_flags
elif [[ "$device_str" == "aarch64"* ]]; then
  IMAGE_TYPE=aarch64
  add_device_flags
else
  fatal "Unsupported device: ${device_str}"
fi

emph "Installing Gimlet Edge Module"

function prompt_deploy_key() {
  read -r -p "Deploy Key: " DEPLOY_KEY
}

if [[ -z "$DEPLOY_KEY" ]]; then
  prompt_deploy_key
fi
while [[ -z "$DEPLOY_KEY" ]]; do
  printf "Must supply a Deploy Key to continue.\n"
  prompt_deploy_key
done

cmdline_opts+=(--deploy_key="$DEPLOY_KEY")
cmdline_opts+=(--controlplane_addr="$CONTROLPLANE_ADDR")

if [[ -z "${IMAGE_TAG}" ]]; then
  IMAGE_TAG="${IMAGE_VERSION}"
  if [[ -n "${IMAGE_TYPE}" ]]; then
    IMAGE_TAG="${IMAGE_TYPE}-${IMAGE_TAG}"
  fi
fi

IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
if [[ -n "${IMAGE_DIGEST}" ]]; then
  IMAGE="${IMAGE}@${IMAGE_DIGEST}"
fi

echo "Running container: $IMAGE"

container_id=$(docker run \
  "${docker_flags[@]}" \
  "$IMAGE" \
  "${cmdline_opts[@]}")

function success() {
  cat <<EOS
${tty_bold}${tty_green}Gimlet has been successfully installed!${tty_reset}
Please visit ${tty_underline}https://${CONTROLPLANE_ADDR}${tty_reset} to deploy your first model.
EOS
  exit 0
}

start_time=$(date +%s)
while true; do
  curr_time=$(date +%s)
  if docker logs "$container_id" 2>&1 | grep -q "Starting GRPC Bridge"; then
    success
  fi
  if [[ $(docker container inspect -f '{{.State.Running}}' "${container_id}") == "false" ]]; then
    docker logs "$container_id"
    fatal "Gimlet Edge Module failed to start."
  fi
  if [[ $((curr_time - start_time)) -gt 120 ]]; then
    docker logs "$container_id"
    fatal "Timed out waiting for Gimlet Edge Module to start."
  fi
  sleep 5
done
