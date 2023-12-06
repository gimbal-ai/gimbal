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

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@gml//bazel:deb_archive.bzl", "DEB_ARCHIVE_ATTRS", "deb_archive_common_impl")

def _remove_leading_dot(path):
    return path.removeprefix("./")

def _is_excluded(rctx, path):
    for exc in rctx.attr.exclude_paths:
        if path.startswith(exc):
            return True
    return False

def _bazel_target_name_from_path(path):
    return path.replace("/", "_")

def _run_find(rctx, find_args, fields, sep = ";;;"):
    res = rctx.execute(
        [
            "find",
            ".",
        ] + find_args + [
            "-printf",
            sep.join(fields) + "\n",
        ],
    )
    if res.return_code != 0:
        fail("Failed to run `find`: {}".format(res.stderr))
    results = []
    for line in res.stdout.splitlines():
        splits = line.split(sep)
        if len(splits) != len(fields):
            fail("Failed to split `find` output by separator ({sep}). Likely that the separator appears in a filename".format(sep = sep))
        if len(fields) == 1:
            results.append(splits[0])
        else:
            results.append(splits)
    return results

def _find_symlinks(rctx, ignore_args):
    results = _run_find(
        rctx,
        find_args = ignore_args + ["-type", "l"],
        fields = ["%p", "%l"],
    )

    symlinks = dict()
    repo_path = str(rctx.path("."))
    for res in results:
        path = res[0]
        target = res[1]
        path = _remove_leading_dot(path)
        if _is_excluded(rctx, path):
            continue
        if target.startswith("/"):
            target = "/" + paths.relativize(target, repo_path)
        symlinks[_bazel_target_name_from_path(path)] = (path, target)
    return symlinks

def _find_empty_directories(rctx, ignore_args):
    results = _run_find(rctx, ["-empty"] + ignore_args + ["-type", "d"], ["%p"])
    return [
        _remove_leading_dot(path)
        for path in results
        if not _is_excluded(rctx, _remove_leading_dot(path))
    ]

def _find_regular_files(rctx, ignore_args):
    results = _run_find(rctx, ignore_args + ["-type", "f"], ["%p", "%#m"])

    mode_to_files = dict()
    for res in results:
        path = res[0]
        mode = res[1]
        path = _remove_leading_dot(path)
        if _is_excluded(rctx, path):
            continue
        if mode not in mode_to_files:
            mode_to_files[mode] = []
        mode_to_files[mode].append(path)
    return mode_to_files

def _pkg_provider_repo_impl(rctx, ignore_paths):
    ignore_find_args = []
    for path in ignore_paths:
        ignore_find_args += [
            "-not",
            "-path",
            "./{}*".format(path),
        ]

    symlinks = _find_symlinks(rctx, ignore_find_args)
    empty_dirs = _find_empty_directories(rctx, ignore_find_args)
    regular_files = _find_regular_files(rctx, ignore_find_args)

    rctx.template(
        "BUILD.bazel",
        rctx.attr._pkg_provider_build_tpl,
        substitutions = {
            "{deps}": str(rctx.attr.deps),
            "{empty_dirs}": str(empty_dirs),
            "{regular_files}": str(regular_files),
            "{symlinks}": str(symlinks),
        },
    )

_PKG_PROVIDER_REPO_ATTRS = dict(
    deps = attr.string_list(default = []),
    exclude_paths = attr.string_list(default = []),
    _pkg_provider_build_tpl = attr.label(
        allow_single_file = True,
        default = Label("@gml//bazel/rules_pkg:pkg_provider.BUILD"),
    ),
)

def _deb_archive_w_pkg_providers_impl(rctx):
    download_dir = deb_archive_common_impl(rctx)
    _pkg_provider_repo_impl(rctx, [download_dir])

deb_archive_w_pkg_providers = repository_rule(
    doc = """Generates a repository with rules_pkg providers for the files in a .deb archive.
    Generates a PackageFilesInfo for each set of regular files that share the same mode octal,
    a PackageSymlinkInfo for each symlink, and PackageDirsInfo for all empty dirs.
    Also, creates a PackageFilegroupInfo providing target called `all_files` in the repo root.
""",
    implementation = _deb_archive_w_pkg_providers_impl,
    attrs = dict(
        DEB_ARCHIVE_ATTRS,
        **_PKG_PROVIDER_REPO_ATTRS
    ),
)
