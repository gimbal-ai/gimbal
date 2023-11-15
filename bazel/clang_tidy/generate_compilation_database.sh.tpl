#!/bin/bash -e

yq="%yq%"
compdb="%out_path%"
hdrs=(%hdrs%)
srcs=(%srcs%)

echo "[]" >"${compdb}"

for path in "${hdrs[@]}"; do
  "${yq}" -i -o=json '. += {"arguments": %hdr_cmdline% + ["-c", "'"${path}"'"], "file": "'"${path}"'"}' "${compdb}"
done
for path in "${srcs[@]}"; do
  "${yq}" -i -o=json '. += {"arguments": %src_cmdline% + ["-c", "'"${path}"'"], "file": "'"${path}"'"}' "${compdb}"
done
