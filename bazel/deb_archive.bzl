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

def _deb_archive_impl(rctx):
    download_dir = "__downloaded_deb"
    if rctx.execute(["mkdir", download_dir]).return_code != 0:
        fail("failed to create temporary download dir")

    deb_file = "{name}.deb".format(name = rctx.attr.name)
    rctx.download(
        url = rctx.attr.urls,
        output = "{download_dir}/{file}".format(
            download_dir = download_dir,
            file = deb_file,
        ),
        sha256 = rctx.attr.sha256,
    )
    ar_command = [
        "ar",
        "x",
        deb_file,
    ]
    if rctx.execute(ar_command, working_directory = download_dir).return_code != 0:
        fail("failed to extract tar files from deb")

    data_archive = "{download_dir}/{file}".format(
        download_dir = download_dir,
        file = rctx.attr.deb_data_archive,
    )
    rctx.extract(
        archive = data_archive,
        output = rctx.attr.prefix,
        stripPrefix = rctx.attr.strip_prefix,
        rename_files = rctx.attr.rename_files,
    )

    bazel_file_content = rctx.attr.build_file_content
    if rctx.attr.build_file != None:
        bazel_file_content = rctx.read(Label(rctx.attr.build_file))
    rctx.file("BUILD.bazel", content = bazel_file_content)

deb_archive = repository_rule(
    implementation = _deb_archive_impl,
    attrs = dict(
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
        build_file_content = attr.string(
            doc = "BUILD.bazel content to add to BUILD.bazel in root of repo. Mutually exclusive with build_file",
        ),
        build_file = attr.string(
            doc = "BUILD.bazel file to add to root of repo. Mutually exclusive with build_content.",
        ),
    ),
)
