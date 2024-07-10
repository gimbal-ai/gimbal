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

load("@rules_pkg//pkg:providers.bzl", "PackageDirsInfo", "PackageFilegroupInfo", "PackageFilesInfo", "PackageSymlinkInfo")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel/cc_toolchains/sysroots:sysroot_toolchain.bzl", "SysrootPathInfo")

# buildifier: disable=unused-variable
def _noop(*args, **kwargs):
    pass

def _handle_rules_pkg_providers(ctx, files_info_cb = _noop, dirs_info_cb = _noop, symlink_info_cb = _noop):
    for src in ctx.attr.srcs:
        if PackageFilesInfo in src:
            files_info_cb(src[PackageFilesInfo])
        if PackageDirsInfo in src:
            dirs_info_cb(src[PackageDirsInfo])
        if PackageSymlinkInfo in src:
            symlink_info_cb(src[PackageSymlinkInfo])
        if PackageFilegroupInfo in src:
            for (files_info, _) in src[PackageFilegroupInfo].pkg_files:
                files_info_cb(files_info)
            for (dirs_info, _) in src[PackageFilegroupInfo].pkg_dirs:
                dirs_info_cb(dirs_info)
            for (symlink_info, _) in src[PackageFilegroupInfo].pkg_symlinks:
                symlink_info_cb(symlink_info)

def _collect_nonempty_dirs(path, non_empty_dirs):
    """Add all parent directories of `path` to `non_empty_dirs`."""
    splits = path.split("/")
    for i in range(len(splits)):
        path = "/".join(splits[:i])
        if path:
            non_empty_dirs[path] = True

def _matches_filters(ctx, destination):
    """Returns if `destination` has a prefix in ctx.attr.path_prefix_filters."""
    if not ctx.attr.path_prefix_filters:
        return True
    for path in ctx.attr.path_prefix_filters:
        if destination.startswith(path):
            return True
    return False

def _create_empty_dir(ctx, directory, out_files, non_empty_dirs):
    if directory in non_empty_dirs:
        return
    if not _matches_filters(ctx, directory):
        return
    _collect_nonempty_dirs(directory, non_empty_dirs)
    out = ctx.actions.declare_directory(directory)
    ctx.actions.run_shell(outputs = [out], command = "mkdir -p {}".format(directory))
    out_files.append(out)

def _symlink_sysroot_files_impl(ctx):
    out_files = []
    transitive_files = []
    non_empty_dirs = dict()
    empty_dirs = []
    sysroot_path = "/".join([ctx.genfiles_dir.path, "external", ctx.label.workspace_name])

    def handle_files_info(files_info):
        for dest, file in files_info.dest_src_map.items():
            if not _matches_filters(ctx, dest):
                continue
            _collect_nonempty_dirs(dest, non_empty_dirs)
            out = ctx.actions.declare_file(dest)
            ctx.actions.symlink(output = out, target_file = file)
            out_files.append(out)
            transitive_files.append(file)

    def handle_dirs_info(dirs_info):
        empty_dirs.extend(dirs_info.dirs)

    def handle_symlink_info(symlink_info):
        if not _matches_filters(ctx, symlink_info.destination):
            return
        _collect_nonempty_dirs(symlink_info.destination, non_empty_dirs)
        out = ctx.actions.declare_symlink(symlink_info.destination)
        target = symlink_info.target
        if target.startswith("/"):
            depth = symlink_info.destination.count("/")
            target = "/".join([".." for i in range(depth)] + [target[1:]])

        ctx.actions.symlink(output = out, target_path = target)
        out_files.append(out)

    _handle_rules_pkg_providers(
        ctx,
        files_info_cb = handle_files_info,
        dirs_info_cb = handle_dirs_info,
        symlink_info_cb = handle_symlink_info,
    )

    # Create empty directories from the deepest to the shallowest to avoid ArtifactPrefixConflictExceptions.
    # Also, only create directories that haven't already been implicitly created by previous files or directories.
    # eg. if there's a file info with: a/b/c.txt and a dir info with: a/b, then a/b shouldn't be created because
    # it was implicitly created by the file, a/b/c.txt. Similarly, if there's two dir infos with a/b/c and a/b,
    # then a/b shouldn't be created because it was implicitly created by a/b/c.
    sorted_empty_dirs = sorted(empty_dirs, key = lambda d: -d.count("/"))
    for d in sorted_empty_dirs:
        _create_empty_dir(ctx, d, out_files, non_empty_dirs)

    return [
        DefaultInfo(
            files = depset(out_files, transitive = [depset(transitive_files)]),
        ),
        SysrootPathInfo(
            path = sysroot_path,
        ),
    ]

_symlink_sysroot_files = rule(
    implementation = _symlink_sysroot_files_impl,
    attrs = dict(
        srcs = attr.label_list(
            doc = """A list of targets providing rules_pkg providers. All provided files will be symlinked into the sysroot at the paths specified by rules_pkg providers.""",
            mandatory = True,
            providers = [
                [PackageFilegroupInfo, DefaultInfo],
                [PackageFilesInfo, DefaultInfo],
                [PackageDirsInfo],
                [PackageSymlinkInfo],
            ],
        ),
        path_prefix_filters = attr.string_list(
            doc = "Filter srcs to only paths with one of the given prefixes. Empty list means no filtering takes place.",
        ),
    ),
)

def _root_cert_impl(ctx):
    cert_files = dict()

    def handle_files_info(files_info):
        for dest, file in files_info.dest_src_map.items():
            if not dest.startswith(ctx.attr.cert_prefix) or not dest.endswith(ctx.attr.cert_suffix):
                continue
            cert_files[file.path] = file

    _handle_rules_pkg_providers(ctx, files_info_cb = handle_files_info)

    out = ctx.actions.declare_file(ctx.attr.name + ".crt")
    inputs = cert_files.values()
    cmd = "cat $@ > {}".format(
        out.path,
    )
    ctx.actions.run_shell(
        outputs = [out],
        inputs = inputs,
        command = cmd,
        arguments = list(cert_files.keys()),
    )
    dest_src_map = dict()
    dest_src_map[ctx.attr.root_cert_path] = out

    return [
        DefaultInfo(files = depset([out])),
        PackageFilesInfo(
            dest_src_map = dest_src_map,
        ),
    ]

_root_cert = rule(
    implementation = _root_cert_impl,
    attrs = dict(
        srcs = attr.label_list(
            doc = """A list of targets providing rules_pkg providers. These targets will be used to generate a root cert from the given cert path.""",
            mandatory = True,
            providers = [
                [PackageFilegroupInfo, DefaultInfo],
                [PackageFilesInfo, DefaultInfo],
                [PackageDirsInfo],
                [PackageSymlinkInfo],
            ],
        ),
        cert_prefix = attr.string(
            doc = "Prefix to match to find certs",
            default = "usr/share/ca-certificates",
        ),
        cert_suffix = attr.string(
            doc = "Suffix to match to find certs.",
            default = ".crt",
        ),
        root_cert_path = attr.string(
            doc = "path to create root cert at.",
            default = "etc/ssl/certs/ca-certificates.crt",
        ),
    ),
)

def create_sysroot(name, srcs, path_prefix_filters = [], **kwargs):
    """Create a sysroot from a list of targets that provide rules_pkg providers.

    This should only be called from inside a repo created by `sysroot_repo`.
    The sysroot files will be symlinked into the current package, and a tar file for the sysroot will be built.
    """
    _root_cert(
        name = name + "_root_cert",
        srcs = srcs,
    )
    srcs.append(":" + name + "_root_cert")

    _symlink_sysroot_files(
        name = name + "_all_files",
        srcs = srcs,
        path_prefix_filters = path_prefix_filters,
        **kwargs
    )

    pkg_tar(
        name = name,
        srcs = srcs,
        **kwargs
    )
