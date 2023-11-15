#!/bin/bash -e
yq="%yq%"
compdb="%compile_commands%"
clang_tidy="%clang_tidy%"

file="$1"
fixes_yaml="$2"

# Since compile_commands.json requires an absolute path for "directory", we add the "directory" right before we run clang tidy.
# If we added it when the compile_commands are generated it would point to an old bazel sandbox.
"${yq}" -o=json '.[].directory="'"$(pwd)"'"' "${compdb}" >"$(pwd)/compile_commands.json"
"${clang_tidy}" --export-fixes="${fixes_yaml}" "${file}"
