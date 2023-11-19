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

common_flags=(
  -d
  --rm
  --network=host
  -v "$GML_CACHE_DIR:/gml"
  -v /usr/lib:/host_lib
)
extra_flags=()

SERIAL_NUMBER="Unknown"

if [[ $(device_type) == "NVIDIA Orin Nano"* ]]; then
  read -r SERIAL_NUMBER </sys/firmware/devicetree/base/serial-number
  extra_flags+=(
    --privileged
    --runtime nvidia
    --gpus all
    -v /tmp/argus_socket:/tmp/argus_socket
    -v /usr/local/cuda:/host_cuda
  )
else
  fatal "NVIDIA Orin Nano devices are the only supported hardware."
fi

emph "Installing Gimlet Edge Module"

function prompt_deploy_key() {
  read -r -p "Deploy Key: " DEPLOY_KEY
}

prompt_deploy_key
while [[ -z "$DEPLOY_KEY" ]]; do
  printf "Must supply a Deploy Key to continue.\n"
  prompt_deploy_key
done

DEFAULT_CONTROL_PLANE="dev.app.gimletlabs.dev"
USE_CONTROL_PLANE=${GML_CONTROL_PLANE:-${DEFAULT_CONTROL_PLANE}}

mkdir -p "$GML_CACHE_DIR"

docker run \
  "${common_flags[@]}" \
  "${extra_flags[@]}" \
  us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/gem_image:jetson-dev-latest \
  --deploy_key="$DEPLOY_KEY" \
  --controlplane_addr="$USE_CONTROL_PLANE:443" \
  --device_serial "$SERIAL_NUMBER"

cat <<EOS
${tty_bold}${tty_green}Gimlet has been succesfully installed!${tty_reset}
Please visit ${tty_underline}https://${USE_CONTROL_PLANE}${tty_reset} to deploy your first model.
EOS
