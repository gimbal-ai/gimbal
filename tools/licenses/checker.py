#!/usr/bin/env python3

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

import argparse
import logging
import os.path
import re
import sys

company_name = "Gimlet Labs, Inc"
copyright_year = "2023"

apache2_license_header = """
Copyright 2023- Gimlet Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-License-Identifier: Apache-2.0
"""

private_license_header = """
Copyright © 2023- Gimlet Labs, Inc.
All Rights Reserved.

NOTICE:  All information contained herein is, and remains
the property of Gimlet Labs, Inc. and its suppliers,
if any.  The intellectual and technical concepts contained
herein are proprietary to Gimlet Labs, Inc. and its suppliers and
may be covered by U.S. and Foreign Patents, patents in process,
and are protected by trade secret or copyright law. Dissemination
of this information or reproduction of this material is strictly
forbidden unless prior written permission is obtained from
Gimlet Labs, Inc.

SPDX-License-Identifier: Proprietary
"""

license_by_spdx = {
    "Apache-2.0": apache2_license_header,
    "Proprietary": private_license_header,
}


def get_spdx_from_license_contents(contents: str):
    if "All Rights Reserved." in contents:
        return "Proprietary"
    if "Apache License, Version 2.0" in contents:
        return "Apache-2.0"
    if "Permission is hereby granted, free of charge" in contents:
        return "MIT"
    if "GNU GENERAL PUBLIC LICENSE" in contents:
        return "GPL-2.0"
    raise Exception("Cannot determine license type")


def get_spdx_from_license(path):
    with open(path) as f:
        contents = f.read()
        return get_spdx_from_license_contents(contents)


def c_style_license_wrapper(txt: str):
    txt = txt.strip()
    lines = [" * " + line for line in txt.split("\n")]
    lines = [line[:-1] if line.endswith(" ") else line for line in lines]
    txt_with_stars = "\n".join(lines)
    return "/*\n" + txt_with_stars + "\n */"


def sh_style_license_wrapper(txt: str):
    txt = txt.strip()
    lines = ["# " + line for line in txt.split("\n")]
    lines = [line[:-1] if line.endswith(" ") else line for line in lines]
    return "\n".join(lines)


def has_spdx(blob: str):
    return (
        ("\n# SPDX-License-Identifier:" in blob)
        or ("\n// SPDX-License-Identifier:" in blob)
        or ("\n * SPDX-License-Identifier:" in blob)
    )


shebang_regex = re.compile(r"^(#!.*)")
# The list of matches, this is an array to avoid an uncertainty with overlapping expression.
# Please keep the list sorted, by language name.
matchers = [
    {
        "name": "c_style",
        "exprs": [
            re.compile(r"^.*\.(cc|cpp|h|hpp|c|inl)$"),
            re.compile(r"^.*\.(bt)$"),
            re.compile(r"^.*\.(java)$"),
            re.compile(r"^.*\.(js|jsx|ts|tsx)$"),
            re.compile(r"^.*\.(proto)$"),
            re.compile(r"^.*\.(css|scss)$"),
        ],
        "wrapper": c_style_license_wrapper,
    },
    {
        "name": "go",
        "exprs": [
            re.compile(r"^.*\.go$"),
        ],
        "wrapper": c_style_license_wrapper,
        "skip_lines": [
            re.compile(r"^(// \+build.*)"),
        ],
    },
    {
        "name": "shell_without_shebang",
        "exprs": [
            # Bazel.
            re.compile(r"^.*\.(bazel|bzl)$"),
            re.compile(r"^.*\.(BUILD)$"),
            re.compile(r"BUILD.bazel$$"),
            # Docker file.
            re.compile(r"^Dockerfile$"),
            re.compile(r"^Dockerfile\..*$"),
            # Makefiles.
            re.compile(r"^Makefile$"),
            # Starlark..
            re.compile(r"^.*\.(sky)$"),
            # Ruby.
            re.compile(r"^.*\.(rb)$"),
            # Terraform.
            re.compile(r"^.*\.(tf)$"),
        ],
        "wrapper": sh_style_license_wrapper,
    },
    {
        "name": "shell_with_shebang",
        "exprs": [
            # Python.
            re.compile(r"^.*\.(py)$"),
            # Shell.
            re.compile(r"^.*\.(sh)$"),
        ],
        "wrapper": sh_style_license_wrapper,
        "skip_lines": [
            shebang_regex,
        ],
    },
]


def is_generated_code(file_path: str):
    return (
        file_path.endswith(".gen.go")
        or file_path.endswith(".pb.go")
        or file_path.endswith(".deepcopy.go")
        or file_path.endswith("/schema.ts")
    )


def is_skipped(file_path: str):
    license_file = file_path in ["LICENSE", "LICENSE.txt", "go.mod", "go.sum"]
    return is_generated_code(file_path) or license_file


def parse_args():
    parser = argparse.ArgumentParser(description="Check/Fix license info in file")
    parser.add_argument(
        "files", nargs="+", type=str, help="the name(s) of the file(s) to check. "
    )
    parser.add_argument(
        "-a",
        required=False,
        action="store_true",
        default=False,
        help="automatically fix the file",
    )
    return parser.parse_args(sys.argv[1:])


def find_matcher(path: str):
    base = os.path.basename(path)
    for m in matchers:
        for e in m["exprs"]:
            if e.match(base) is not None:
                return m
    return None


class AddLicenseDiff:
    def __init__(self, filename, start_line, txt):
        self._filename = filename
        self._start_line = start_line
        self._txt = txt

    def stringify(self):
        # <Filename>:<LineNumber>,<offset>
        s = "{}:{},1\n".format(self._filename, self._start_line)
        s += "<<<<<\n"
        s += "=====\n"
        s += self._txt + "\n"
        s += ">>>>>"
        return s

    def fix(self, filepath):
        file_lines = None
        with open(filepath, "r") as f:
            file_lines = f.readlines()

        with open(filepath, "w") as f:
            if len(file_lines) == 0:
                f.write(self._txt + "\n")
            for idx, l in enumerate(file_lines):
                # The line is 1 indexed.
                if (idx + 1) == self._start_line:
                    f.write(self._txt + "\n")
                f.write(l)


def generate_modifications_diff_if_needed(path, contents):
    # If the license is Apache-2.0, we should accept the license if the copyright already
    # belongs to the company. Alternatively, we should accept the license if we have a
    # modified copyright.
    content_lines = contents.split("\n")
    offset = 0
    original_copyright = ""
    copyright_comment = ""

    for line in content_lines:
        offset += 1
        result = re.match("(#|\s\*|\/\/)? Copyright \d\d\d\d- ([\w\s]+).", line)
        if result is not None:
            copyright_comment = result.group(1)
            copyright_owner = result.group(2)
            original_copyright = copyright_owner
            break

    if original_copyright == "":
        # TODO: License is malformed, there is no copyright line.
        return None
    elif original_copyright == company_name:
        return None

    # Check that the next line is the Modifications copyright, and add it if not.
    result = re.match(
        "(#|\s\*|\/\/)? Modifications Copyright \d\d\d\d- ([\w\s]+).",
        content_lines[offset],
    )
    if result is None:
        license_text = "{0} Modifications Copyright {1}- {2}.".format(
            copyright_comment, copyright_year, company_name
        )
        return AddLicenseDiff(path, offset + 1, license_text)
    if result.group(2) != company_name:
        # TODO: Modifications copyright is written to another author.
        return None

    return None


def generate_diff_if_needed(path):
    if is_skipped(path):
        return None

    matcher = find_matcher(path)
    if not matcher:
        logging.error("Did not find valid matcher for file: {}".format(path))
        return None

    # Read the file contents.
    contents = None
    with open(path, "r") as f:
        contents = f.read()

    # If the file already has SPDX, we usually just skip it, unless
    # it is an Apache 2.0 license from another open source project.
    if has_spdx(contents):
        license = get_spdx_from_license_contents(contents)
        if license == "Apache-2.0":
            return generate_modifications_diff_if_needed(path, contents)
        return None

    # Keep popping up directories util we find a LICENSE file.
    # We will use that as the default license for this file.
    (subpath, remain) = os.path.split(path)
    license = None
    while True:
        if remain == "":
            break

        license_file = os.path.join(subpath, "LICENSE")
        if os.path.isfile(license_file):
            license = get_spdx_from_license(license_file)
            break
        (subpath, remain) = os.path.split(subpath)

    # Some files have lines that need to occur before the license. This includes
    # go build constraints, and shell lines.
    content_lines = contents.split("\n")
    offset = 0

    if "skip_lines" in matcher:
        for s in matcher["skip_lines"]:
            if s.match(content_lines[offset]) and offset < len(content_lines):
                offset += 1

    # We check and make sure the exact copy of the license occurs after skipping the offset.
    contents_with_offset = "\n".join(content_lines[offset + 1 :])
    expected_license = matcher["wrapper"](license_by_spdx[license])
    if not contents_with_offset.startswith(expected_license):
        license_text = expected_license
        # If we aren't the first line then add a space between the license and other lines.
        # Since we split, '' means there was a \n.
        if offset > 0:
            if content_lines[offset - 1] != "":
                license_text = "\n" + license_text
        if offset < len(content_lines):
            if content_lines[offset] != "":
                license_text = license_text + "\n"
        return AddLicenseDiff(path, offset + 1, license_text)


def main():
    args = parse_args()

    for path in args.files:
        autofix = args.a
        if os.path.isfile(path):
            diff = generate_diff_if_needed(path)
            if diff is not None:
                if autofix:
                    diff.fix(path)
                else:
                    print(diff.stringify())
                    sys.exit(1)
        else:
            logging.fatal("each <file> argument needs to be a file")


if __name__ == "__main__":
    main()
