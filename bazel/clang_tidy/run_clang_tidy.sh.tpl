#!/bin/bash -e
yq="%yq%"
compdb="%compile_commands%"
clang_tidy="%clang_tidy%"

file="$1"
fixes_yaml="$2"

# If the fixes file isn't created, bazel complains since the output is expected to be created.
# So preemptively touch the file in case there are no suggested fixes from clang-tidy.
touch "${fixes_yaml}"
# Since compile_commands.json requires an absolute path for "directory", we add the "directory" right before we run clang tidy.
# If we added it when the compile_commands are generated it would point to an old bazel sandbox.
"${yq}" -o=json '.[].directory="'"$(pwd)"'"' "${compdb}" >"$(pwd)/compile_commands.json"
# We use -fno-caret-diagnostics to silence the output some more. See https://github.com/llvm/llvm-project/issues/47042
"${clang_tidy}" --quiet --extra-arg="-fno-caret-diagnostics" --export-fixes="${fixes_yaml}" "${file}"
