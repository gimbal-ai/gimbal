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

trap exit INT

if [ "$#" -lt 4 ]; then
  echo "Usage: build_tar_for_packages.sh <package_satisifier_path> <yaml_base_dir> <package_database_file> <output_tar_path> <variant> [<features>...]"
  echo -e "\t This script downloads all of the debs (along with depedencies) it finds in the yamls for the variant and features"
  echo -e "\t and extracts them into a single 'sysroot'"
  echo -e "\t The 'sysroot' is then tar'd and output at <output_tar_path>"
  exit 1
fi

package_satisifier_path="$(realpath "$1")"
yaml_base_dir="$(realpath "$2")"
package_database_file="$(realpath "$3")"
output_tar_path="$(realpath "$4")"
variant="$5"
features=("default" "${@:6}")

base_paths=()

case "$variant" in
runtime | build | test)
  base_paths+=("runtime.yaml")
  ;;&
build | test)
  base_paths+=("build.yaml")
  ;;&
test)
  base_paths+=("test.yaml")
  ;;&
esac

if [[ "${#base_paths[@]}" == 0 ]]; then
  echo "sysroot variant must be one of runtime,build,test"
  exit 2
fi

yamls=()
for feat in "${features[@]}"; do
  feat_dir="${yaml_base_dir}/${feat}"
  if [[ ! -d "${feat_dir}" ]]; then
    echo "unknown feature '${feat}': expected directory ${feat_dir} to exist"
    exit 2
  fi
  feat_yamls=()
  for base in "${base_paths[@]}"; do
    f="${feat_dir}/${base}"
    if [[ -f "$f" ]]; then
      feat_yamls+=("$f")
    fi
  done

  if [[ "${#feat_yamls[@]}" == 0 ]]; then
    echo "feature '${feat}' didn't match any yaml paths for variant '${variant}'"
    exit 2
  fi

  yamls+=("${feat_yamls[@]}")
done

package_parser_args=("--pkgdb" "${package_database_file}")
for yaml in "${yamls[@]}"; do
  package_parser_args+=("--specs" "${yaml}")
done

debian_mirror="${DEBIAN_MIRROR:-http://ftp.us.debian.org/debian/}"
debs=()
while read -r deb; do
  debs+=("${debian_mirror}/${deb}")
done < <("${package_satisifier_path}" "${package_parser_args[@]}")

echo "Dependencies to be added to archive:"
for deb in "${debs[@]}"; do
  echo "- ${deb}"
done

declare -A paths_to_exclude
while read -r path; do
  if [ -n "${path}" ]; then
    paths_to_exclude["${path}"]=true
  fi
done < <(yq eval -N '.path_excludes[]' "${yamls[@]}")

declare -A extra_dirs
while read -r dir; do
  if [ -n "${dir}" ]; then
    extra_dirs["${dir}"]=true
  fi
done < <(yq eval -N '.extra_dirs[]' "${yamls[@]}")

relativize_symlinks() {
  dir="$1"
  libdirs=("lib" "lib64" "usr/lib")
  pushd "${dir}" >/dev/null

  while read -r link target; do
    # Skip links targeting non-absolute paths.
    if [[ "${target}" != "/"* ]]; then
      continue
    fi
    # Remove all non-"/" characters from the link name. Then replace each "/" with "../".
    prefix=$(echo "${link}" | sed -e 's|[^/]||g' | sed -e 's|/|../|g')
    ln -snf "${prefix}${target}" "${link}"
  done < <(find "${libdirs[@]}" -type l -printf '%p %l\n')

  popd >/dev/null
}

create_root_cert() {
  root_dir="$1"
  combined_certs="$(find "${root_dir}/usr/share/ca-certificates" -type f -name '*.crt' -exec cat {} +)"
  if [ -n "${combined_certs}" ]; then
    # Only create the root cert file if there were certificates in the ca-certificates directory.
    echo "${combined_certs}" >"${root_dir}/etc/ssl/certs/ca-certificates.crt"
  fi
}

inside_tmpdir() {
  echo "${debs[@]}" | xargs curl -fLO --remote-name-all &>/dev/null

  root_dir="root"
  while read -r deb; do
    dpkg-deb -x "${deb}" "${root_dir}" &>/dev/null
  done < <(ls -- *.deb)

  create_root_cert "${root_dir}"

  for dir in "${!extra_dirs[@]}"; do
    mkdir -p "${root_dir}/${dir}"
  done

  for path in "${!paths_to_exclude[@]}"; do
    echo "Removing ${path} from sysroot"
    rm -rf "${root_dir:?}/${path:?}"
  done

  relativize_symlinks "${root_dir}"

  # Pick a deterministic mtime so that the sha sums only change if there are actual changes to the sysroot.
  tar --mtime="2023-01-01 00:00:00 UTC" -C "${root_dir}" -czf "${output_tar_path}" .
}

tmpdir="$(mktemp -d)"
pushd "${tmpdir}" >/dev/null

inside_tmpdir

popd >/dev/null
rm -rf "${tmpdir}"
