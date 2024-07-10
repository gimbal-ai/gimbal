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

def _deb_archive_common_impl(rctx):
    download_dir = "__downloaded_deb"
    rctx.download_and_extract(
        url = rctx.attr.urls,
        output = download_dir,
        sha256 = rctx.attr.sha256,
    )

    res = rctx.execute(
        [
            "find",
            ".",
            "-type",
            "f",
            "-name",
            "data.tar*",
        ],
        working_directory = download_dir,
    )
    data_tar_name = res.stdout.strip("\n")

    data_archive = "{download_dir}/{file}".format(
        download_dir = download_dir,
        file = data_tar_name,
    )
    rctx.extract(
        archive = data_archive,
        output = rctx.attr.prefix,
        stripPrefix = rctx.attr.strip_prefix,
        rename_files = rctx.attr.rename_files,
    )
    return download_dir

def _deb_archive_impl(rctx):
    _deb_archive_common_impl(rctx)

    bazel_file_content = rctx.attr.build_file_content
    if rctx.attr.build_file != None:
        bazel_file_content = rctx.read(Label(rctx.attr.build_file))
    rctx.file("BUILD.bazel", content = bazel_file_content)

DEB_ARCHIVE_ATTRS = dict(
    urls = attr.string_list(
        doc = "URLs to download .deb file from",
    ),
    sha256 = attr.string(
        doc = "SHA256 checksum of .deb file",
    ),
    strip_prefix = attr.string(
        doc = "Prefix to strip during extraction of data tar inside deb",
    ),
    prefix = attr.string(
        doc = "Prefix to add to paths extracted from data tar",
    ),
    rename_files = attr.string_dict(
        default = dict(),
        doc = "Map of renames for files inside data tar",
    ),
    deb_data_archive = attr.string(
        default = "data.tar.zst",
        doc = "Name of data tar inside deb file",
    ),
)

deb_archive = repository_rule(
    implementation = _deb_archive_impl,
    attrs = dict(
        build_file_content = attr.string(
            doc = "BUILD.bazel content to add to BUILD.bazel in root of repo. Mutually exclusive with build_file",
        ),
        build_file = attr.string(
            doc = "BUILD.bazel file to add to root of repo. Mutually exclusive with build_content.",
        ),
        **DEB_ARCHIVE_ATTRS
    ),
)

deb_archive_common_impl = _deb_archive_common_impl
