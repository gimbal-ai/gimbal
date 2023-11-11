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
sysroot_download_url="$3"
cache_dir="$(realpath "$4")"

tot="$(git rev-parse --show-toplevel)"

# Sysroots to produce in 'arch variant libc_version feat1 feat2' format.
sysroots=(
  "aarch64 runtime glibc2_36 default"
  "aarch64 build glibc2_36 default"
  "aarch64 test glibc2_36 default"
  "aarch64 test glibc2_36 default debug"
  "x86_64 runtime glibc2_36 default"
  "x86_64 build glibc2_36 default"
  "x86_64 test glibc2_36 default"
  "x86_64 test glibc2_36 default debug"
)

# collect features that are enabled for each variant in any sysroot.
# This is used to ensure the featureless versions of the variants are not used
# when a feature is enabled.
declare -A variant_features
for config in "${sysroots[@]}"; do
  read -ra config_arr <<<"$config"
  arch="${config_arr[0]}"
  variant="${config_arr[1]}"
  features=("${config_arr[@]:3}")

  var_feats="${variant_features["${variant}"]}"
  for feat in "${features[@]}"; do
    if [[ "${feat}" == "default"* ]]; then
      continue
    fi
    if ! grep "${feat}" < <(echo "${var_feats}") &>/dev/null; then
      if [[ -n "${var_feats}" ]]; then
        var_feats="${var_feats} "
      fi
      var_feats="${var_feats}${feat}"
    fi
  done
  variant_features["${variant}"]="${var_feats}"
done

sysroot_filename() {
  arch="$1"
  variant="$2"
  libc_version="$3"
  features=("${@:4}")
  fname="sysroot-${arch}-${libc_version}-${variant}"
  for feat in "${features[@]}"; do
    if [[ "${feat}" == "default"* ]]; then
      continue
    fi
    fname="${fname}-${feat}"
  done
  fname="${fname}.tar.gz"
  echo "${fname}"
}

sysroot_name() {
  arch="$1"
  variant="$2"
  libc_version="$3"
  features=("${@:4}")
  name="sysroot_${arch}_${libc_version}_${variant}"
  for feat in "${features[@]}"; do
    if [[ "${feat}" == "default"* ]]; then
      continue
    fi
    name="${name}_${feat}"
  done
  echo "${name}"
}

build_sysroot() {
  arch="$1"
  variant="$2"
  libc_version="$3"
  features=("${@:4}")
  fname="$(sysroot_filename "$@")"
  echo "Building ${output_dir}/${fname}"
  docker run -it -v "${output_dir}":/build \
    -v "${cache_dir}":/cache \
    "${docker_image_tag}" \
    "/cache" \
    "/build/${fname}" \
    "${arch}" \
    "${variant}" \
    "${features[@]}"
}

create_sysroots_bzl="${tot}/bazel/cc_toolchains/sysroots/create_sysroots.bzl"
register_sysroots_bzl="${tot}/bazel/cc_toolchains/sysroots/register_sysroots.bzl"
settings_bzl="${tot}/bazel/cc_toolchains/sysroots/settings.bzl"
bazelrc="${tot}/bazel/cc_toolchains/sysroots/.bazelrc"

cat <<EOF >"${create_sysroots_bzl}"
# THIS FILE IS GENERATED BY build_all_sysroots

load("//bazel/cc_toolchains/sysroots:sysroots.bzl", "sysroot_repo")

def _create_sysroots():
EOF

cat <<EOF >"${register_sysroots_bzl}"
# THIS FILE IS GENERATED BY build_all_sysroots

EOF

cat <<EOF >"${settings_bzl}"
# THIS FILE IS GENERATED BY build_all_sysroots

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")


def _sysroot_settings():
EOF

cat <<EOF >"${bazelrc}"
# THIS FILE IS GENERATED BY build_all_sysroots

EOF

sysroot_names="$(mktemp)"

gen_bzl() {
  arch="$1"
  variant="$2"
  libc_version="$3"
  features=("${@:4}")

  name="$(sysroot_name "$@")"
  fname="$(sysroot_filename "$@")"
  sha="$(sha256sum "${output_dir}/${fname}" | awk '{print $1}')"

  feat_list=""
  for feat in "${features[@]}"; do
    if [[ "${feat}" == "default"* ]]; then
      continue
    fi
    if [[ -n "${feat_list}" ]]; then
      feat_list="${feat_list}, "
    fi
    feat_list="${feat_list}\"${feat}\""
  done
  disabled_for_features=""
  for feat in ${variant_features["${variant}"]}; do
    found="false"
    for f in "${features[@]}"; do
      if [[ "$f" == "${feat}" ]]; then
        found="true"
      fi
    done
    if [[ "${found}" != "true" ]]; then
      if [[ -n "${disabled_for_features}" ]]; then
        disabled_for_features="${disabled_for_features}, "
      fi
      disabled_for_features="${disabled_for_features}\"${feat}\""
    fi
  done

  cat <<EOF >>"${create_sysroots_bzl}"
    sysroot_repo(
        name = "${name}",
        target_arch = "${arch}",
        variant = "${variant}",
        libc_version = "${libc_version}",
        sha256 = "${sha}",
        urls = ["${sysroot_download_url}/${fname}"],
        sysroot_features = [${feat_list}],
        disabled_for_features = [${disabled_for_features}],
    )
EOF

  echo "${name}" >>"${sysroot_names}"
}

declare -A uniq_features

for config in "${sysroots[@]}"; do
  read -ra config_arr <<<"$config"

  arch="${config_arr[0]}"
  variant="${config_arr[1]}"
  libc_version="${config_arr[2]}"
  features=("${config_arr[@]:3}")
  build_sysroot "${config_arr[@]}"
  gen_bzl "${config_arr[@]}"
  for feat in "${features[@]}"; do
    if [[ "${feat}" == "default"* ]]; then
      continue
    fi
    uniq_features["${feat}"]=true
  done
done

# Make sure the register loads are sorted.
register_counter=0
while read -r name; do
  cat <<EOF >>"${register_sysroots_bzl}"
load("@${name}//:register_toolchain.bzl", _register${register_counter} = "register_toolchain")
EOF
  register_counter="$((register_counter + 1))"
done < <(sort "${sysroot_names}")

cat <<EOF >>"${register_sysroots_bzl}"

def _register_sysroots():
EOF

register_counter=0
while read -r name; do
  cat <<EOF >>"${register_sysroots_bzl}"
    _register${register_counter}()
EOF
  register_counter="$((register_counter + 1))"
done < <(sort "${sysroot_names}")

for feat in "${!uniq_features[@]}"; do
  cat <<EOF >>"${settings_bzl}"
    bool_flag(
        name = "sysroot_enable_${feat}",
        build_setting_default = False,
    )
    native.config_setting(
        name = "sysroot_${feat}_enabled",
        flag_values = {
            ":sysroot_enable_${feat}": "True",
        },
    )
    native.config_setting(
        name = "sysroot_${feat}_disabled",
        flag_values = {
            ":sysroot_enable_${feat}": "False",
        },
    )
EOF
  cat <<EOF >>"${bazelrc}"
common:enable_sysroot_${feat} --repo_env GML_ENABLE_SYSROOT_${feat^^}=1
common:enable_sysroot_${feat} --//bazel/cc_toolchains/sysroots:sysroot_enable_${feat}
EOF
done

echo "create_sysroots = _create_sysroots" >>"${create_sysroots_bzl}"

echo "register_sysroots = _register_sysroots" >>"${register_sysroots_bzl}"

echo "sysroot_settings = _sysroot_settings" >>"${settings_bzl}"
