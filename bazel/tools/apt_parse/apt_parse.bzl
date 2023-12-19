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

load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("//bazel:gml_build_system.bzl", "no_sysroot")

def _target_to_files(target):
    return [
        f.path
        for f in target[DefaultInfo].files.to_list()
    ]

def _apt_parse_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.bzl_file)
    args = ctx.actions.args()
    args.add("--out_bzl", out)
    args.add_joined("--spec", ctx.attr.specs, join_with = ",", map_each = _target_to_files)
    args.add_joined("--arch", ctx.attr.archs, join_with = ",")
    args.add("--fake_mirroring", ctx.attr.fake_mirroring)

    depsets = []
    for spec in ctx.attr.specs:
        depsets.append(spec[DefaultInfo].files)

    ctx.actions.run(
        executable = ctx.attr._apt_parse[DefaultInfo].files.to_list()[0],
        outputs = [out],
        inputs = depset(transitive = depsets),
        tools = ctx.attr._apt_parse[DefaultInfo].files,
        arguments = [args],
    )

    return DefaultInfo(
        files = depset([out]),
    )

_apt_parse_rule = rule(
    implementation = _apt_parse_impl,
    attrs = dict(
        archs = attr.string_list(
            mandatory = True,
            doc = "A list of architectures to generate deb repos for",
        ),
        bzl_file = attr.string(
            mandatory = True,
            doc = "Name of .bzl file to output",
        ),
        specs = attr.label_list(
            mandatory = True,
            doc = "A list of apt_parse specs to parse.",
            allow_files = True,
        ),
        fake_mirroring = attr.bool(
            doc = "If true, skips mirroring but still generates the deb repos as if mirroring was enabled.",
            default = False,
        ),
        _apt_parse = attr.label(
            default = Label("//bazel/tools/apt_parse"),
            cfg = "exec",
        ),
    ),
)

def apt_parse(name, specs, archs, **kwargs):
    """apt_parse parses a list of apt spec yamls into a bazel macro with all repository_rules for all transitive dependencies.
    This macros creates a `<name>.update` target that updates the `<name>.bzl` file based on the specs.
    It also creates a `<name>.test` target that tests that the `<name>.bzl` file is up-to-date for the specs.
    """
    tags = kwargs.pop("tags", [])
    bzl_file = name + ".bzl"
    _apt_parse_rule(
        name = name,
        archs = archs,
        bzl_file = bzl_file + ".gen",
        specs = specs,
        tags = tags + ["manual"],
        **kwargs
    )

    write_file(
        name = name + "_update_script",
        out = name + "_update.sh",
        content = [
            "#!/bin/bash",
            "cd $BUILD_WORKSPACE_DIRECTORY",
            "cp -fv bazel-bin/{pkg}/{gen_filename} {pkg}/{filename}".format(
                pkg = native.package_name(),
                gen_filename = bzl_file + ".gen",
                filename = bzl_file,
            ),
        ],
        tags = tags + ["manual"],
        **kwargs
    )

    native.sh_binary(
        name = name + ".update",
        srcs = [":" + name + "_update.sh"],
        data = [":" + name],
        tags = tags + ["manual"],
        **kwargs
    )

    _apt_parse_rule(
        name = name + ".test.gen",
        archs = archs,
        bzl_file = bzl_file + ".test.gen",
        specs = specs,
        tags = tags + ["manual"],
        fake_mirroring = True,
        **kwargs
    )

    diff_test(
        name = name + ".test",
        failure_message = "Please run: bazel run //{pkg}:{target}".format(
            pkg = native.package_name(),
            target = name + ".update",
        ),
        file1 = bzl_file,
        file2 = name + ".test.gen",
        target_compatible_with = no_sysroot(),
        tags = tags,
        **kwargs
    )
