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

import argparse
import shutil
from pathlib import Path

import lief


def add_runpath(file_path, output_file_path, new_runpaths):
    elf = lief.parse(file_path)

    modified = False
    for entry in elf.dynamic_entries:
        if isinstance(entry, lief.ELF.DynamicEntryRunPath):
            for path in reversed(new_runpaths):
                entry.insert(0, path)
            modified = True

    if not modified:
        elf.add(lief.ELF.DynamicEntryRunPath(new_runpaths))

    elf.write(str(output_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_libs", required=True, nargs="+", type=Path)
    parser.add_argument("--output_paths", required=True, nargs="+", type=Path)
    parser.add_argument("--runpaths", required=True, nargs="+")

    args = parser.parse_args()

    for lib, out, runpath in zip(args.input_libs, args.output_paths, args.runpaths):
        if runpath == "":
            shutil.copy(lib, out)
            continue
        new_runpaths = runpath.split(":")
        add_runpath(lib, out, new_runpaths)
