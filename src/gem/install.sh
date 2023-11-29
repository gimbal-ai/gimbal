#!/usr/bin/env bash

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

################################################################
# Gimlet Installer
################################################################
set -u

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
tty_cyan="$(tty_escape "36")"

function emph() {
  printf "${tty_bold}${tty_cyan}%s${tty_reset}\n" "$1"
}

function fatal() {
  printf "${tty_bold}${tty_red}%s${tty_reset}\n" "$1"
  exit 1
}

function device_type() {
  if [[ "$(uname -m)" == "x86_64" ]]; then
    echo "x86_64 Generic"
    return
  fi
  if [[ "$(uname -r)" != *"-tegra" ]]; then
    echo "Unknown"
    return
  fi

  if [[ ! -f /proc/device-tree/model ]]; then
    echo "Unknown"
    return
  fi

  read -r MODEL </proc/device-tree/model
  if [[ "$MODEL" == "NVIDIA Orin Nano"* ]]; then
    echo "NVIDIA Orin Nano"
    return
  fi

  echo "Unknown"
  return
}

GML_CACHE_DIR=${GML_CACHE_DIR:-"$HOME/.cache/gml"}

common_docker_flags=(
  -d
  --rm
  --network=host
  -v "$GML_CACHE_DIR:/gml"
  -v /usr/lib:/host_lib
  # Mount /sys/class/net so that GEM can use the mac address as the SERIAL_NUMBER.
  -v /sys:/host/sys
)
extra_docker_flags=()

DEFAULT_IMAGE_TAG=""

cmdline_opts=(
  "--blob_store_dir" "/gml"
  "--sys_class_net_path" "/host/sys/class/net"
)

if [[ "$(device_type)" == "NVIDIA Orin Nano"* ]]; then
  extra_docker_flags+=(
    --privileged
    --runtime nvidia
    --gpus all
    -v /tmp/argus_socket:/tmp/argus_socket
    -v /usr/local/cuda:/host_cuda
  )
  DEFAULT_IMAGE_TAG=jetson-dev-latest
elif [[ "$(device_type)" == "x86_64 Generic" ]]; then
  DEFAULT_IMAGE_TAG=dev-latest
  for vid in "/dev/video"*; do
    extra_docker_flags+=("--device" "${vid}")
  done
else
  fatal "Only NVIDIA Orin Nano devices or x86_64 machines are supported."
fi

emph "Installing Gimlet Edge Module"

function prompt_deploy_key() {
  read -r -p "Deploy Key: " DEPLOY_KEY
}

DEPLOY_KEY=${GML_DEPLOY_KEY:-""}
if [[ -z "$DEPLOY_KEY" ]]; then
  prompt_deploy_key
fi
while [[ -z "$DEPLOY_KEY" ]]; do
  printf "Must supply a Deploy Key to continue.\n"
  prompt_deploy_key
done

cmdline_opts+=(--deploy_key="$DEPLOY_KEY")

DEFAULT_IMAGE_REPO="us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/gem_image"
DEFAULT_CONTROL_PLANE="dev.app.gimletlabs.dev"
USE_CONTROL_PLANE=${GML_CONTROL_PLANE:-${DEFAULT_CONTROL_PLANE}}
cmdline_opts+=(--controlplane_addr="$USE_CONTROL_PLANE:443")

mkdir -p "$GML_CACHE_DIR"

IMAGE_TAG=${GML_IMAGE_TAG:-${DEFAULT_IMAGE_TAG}}
IMAGE_REPO=${GML_IMAGE_REPO:-${DEFAULT_IMAGE_REPO}}
echo "Running container: $IMAGE_REPO:$IMAGE_TAG"

docker run \
  "${common_docker_flags[@]}" \
  "${extra_docker_flags[@]}" \
  "$IMAGE_REPO:$IMAGE_TAG" \
  "${cmdline_opts[@]}"

cat <<EOS
${tty_bold}${tty_green}Gimlet has been succesfully installed!${tty_reset}
Please visit ${tty_underline}https://${USE_CONTROL_PLANE}${tty_reset} to deploy your first model.
EOS
