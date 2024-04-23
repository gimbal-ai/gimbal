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

load("@aspect_bazel_lib//lib:expand_template.bzl", "expand_template")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_image_index", "oci_push", "oci_tarball")
load("@rules_pkg//pkg:providers.bzl", "PackageFilegroupInfo", "PackageFilesInfo", "PackageSymlinkInfo")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel:gml_build_system.bzl", "glibc2_36")
load("//bazel:lib.bzl", "default_arg")
load("//bazel:toolchain_transitions.bzl", "oci_image_arm64", "oci_image_x86_64")

def _gml_oci_image(name, multiarch = False, **kwargs):
    testonly = kwargs.pop("testonly", False)

    if "base" not in kwargs:
        default_arg(kwargs, "os", "linux")
        default_arg(kwargs, "architecture", "amd64")

    image_name = name
    if native.package_name():
        image_name = native.package_name() + "/" + image_name

    oci_image(
        name = name,
        testonly = testonly,
        **kwargs
    )

    oci_tarball(
        name = name + ".tar",
        image = ":" + name,
        repo_tags = [image_name + ":latest"],
        tags = ["manual"],
        testonly = testonly,
    )

    if not multiarch:
        return

    default_arg(kwargs, "tags", [])
    kwargs["tags"] = kwargs["tags"] + ["manual"]

    x86_kwargs = dict(**kwargs)
    if "base" in x86_kwargs:
        x86_kwargs["base"] = x86_kwargs["base"] + "_x86_64"
    else:
        x86_kwargs["architecture"] = "amd64"

    arm64_kwargs = dict(**kwargs)
    if "base" in arm64_kwargs:
        arm64_kwargs["base"] = arm64_kwargs["base"] + "_arm64"
    else:
        arm64_kwargs["architecture"] = "arm64"

    oci_image_x86_64(
        name = name + "_x86_64",
        target_compatible_with = glibc2_36(),
        **x86_kwargs
    )
    oci_image_arm64(
        name = name + "_arm64",
        target_compatible_with = glibc2_36(),
        **arm64_kwargs
    )

    oci_image_index(
        name = name + ".multiarch",
        images = [
            ":" + name + "_x86_64",
            ":" + name + "_arm64",
        ],
        tags = ["manual"],
    )

def _gml_binary_image(name, binary, multiarch = False, **kwargs):
    testonly = kwargs.pop("testonly", False)
    include_runfiles = kwargs.pop("include_runfiles", False)

    default_arg(kwargs, "base", "//:cc_base_image")
    default_arg(kwargs, "tars", [])
    default_arg(kwargs, "entrypoint", ["/app/" + Label(binary).name])

    tar_srcs = [binary]
    if include_runfiles:
        denylist = kwargs.pop("runfiles_denylist", [])
        _collect_runfiles(
            name = name + "_binary_runfiles",
            binaries = [binary],
            testonly = testonly,
            denylist = denylist,
        )
        tar_srcs.append(":" + name + "_binary_runfiles")

    pkg_tar(
        name = name + "_binary_tar",
        srcs = tar_srcs,
        package_dir = "/app",
        testonly = testonly,
    )

    kwargs["tars"] = [name + "_binary_tar"] + kwargs["tars"]
    _gml_oci_image(name, multiarch = multiarch, testonly = testonly, **kwargs)

def _gml_oci_push(name, **kwargs):
    tags = kwargs.pop("remote_tags", [])

    write_file(
        name = name + "_tags_tmpl",
        out = name + "_tags.txt.tmpl",
        content = select({
            "//bazel:stamped": [
                "SYSROOT_PREFIXBUILD_USER",
                "SYSROOT_PREFIXCOMMIT_SHA",
                "SYSROOT_PREFIXTAG",
            ],
            "//conditions:default": [],
        }) + ["SYSROOT_PREFIX" + tag for tag in tags],
    )

    expand_template(
        name = name + "_tags",
        out = name + "_tags.txt",
        stamp_substitutions = {
            "BUILD_USER": "{{STABLE_BUILT_BY}}",
            "COMMIT_SHA": "{{STABLE_BUILD_SCM_REVISION}}",
            "TAG": "{{STABLE_BUILD_TAG}}",
        },
        substitutions = select({
            "//bazel/cc_toolchains/sysroots:sysroot_type_cuda": {"SYSROOT_PREFIX": "do-not-distribute-"},
            "//bazel/cc_toolchains/sysroots:sysroot_type_debian12": {"SYSROOT_PREFIX": ""},
            "//bazel/cc_toolchains/sysroots:sysroot_type_intelgpu": {"SYSROOT_PREFIX": "intelgpu-"},
            "//bazel/cc_toolchains/sysroots:sysroot_type_jetson": {"SYSROOT_PREFIX": "jetson-"},
            "//conditions:default": {"SYSROOT_PREFIX": ""},
        }),
        template = name + "_tags_tmpl",
    )

    oci_push(name = name, remote_tags = name + "_tags", **kwargs)

def _collect_runfiles_impl(ctx):
    dest_to_src = dict()
    external_symlinks = dict()
    for binary in ctx.attr.binaries:
        binary_file = binary[DefaultInfo].files_to_run.executable
        binary_name = binary_file.basename
        runfiles_dir = binary_name + ".runfiles"

        runfiles = []
        if binary[DefaultInfo].default_runfiles:
            runfiles = binary[DefaultInfo].default_runfiles.files.to_list()
        for f in runfiles:
            denied = False
            for partial in ctx.attr.denylist:
                if partial in f.path:
                    denied = True
            if denied:
                continue
            if f.path.startswith("external/sysroot"):
                # Ignore sysroot runfiles.
                continue
            if f.path.find("bazel/test_runners") != -1:
                # Ignore test runners for image runfiles.
                continue
            if f.path.endswith(binary_name):
                # Ignore the binary itself
                continue
            path = f.path
            if path.startswith("bazel-out"):
                _, _, path = path.partition("/bin/")
            if path.startswith("external/"):
                path_wo_external = path.removeprefix("external/")
                repo, _, _ = path_wo_external.partition("/")
                if repo not in external_symlinks:
                    external_symlinks[repo] = PackageSymlinkInfo(
                        destination = "/".join([runfiles_dir, repo]),
                        target = "/".join(["_main/external", repo]),
                    )
            path = "/".join([runfiles_dir, "_main", path])
            dest_to_src[path] = f

    origin = ctx.label
    return [
        PackageFilegroupInfo(
            pkg_files = [
                (
                    PackageFilesInfo(
                        dest_src_map = dest_to_src,
                        attributes = {},
                    ),
                    origin,
                ),
            ],
            pkg_symlinks = [
                (sym_info, origin)
                for sym_info in external_symlinks.values()
            ],
        ),
        DefaultInfo(
            files = depset(dest_to_src.values()),
        ),
    ]

_collect_runfiles = rule(
    implementation = _collect_runfiles_impl,
    attrs = dict(
        binaries = attr.label_list(),
        denylist = attr.string_list(),
    ),
)

def _gml_fast_py_image(name, binary, **kwargs):
    default_base_image = "@gml//bazel/python:python_experimental_base_image" if native.package_name().startswith("src/experimental/") else "@gml//bazel/python:python_base_image"

    default_arg(kwargs, "base", default_base_image)
    default_arg(kwargs, "runfiles_denylist", [])
    default_arg(kwargs, "tars", [])

    binary_name = Label(binary).name
    default_arg(kwargs, "entrypoint", ["python3", "/app/{}.py".format(binary_name)])
    python_path = [
        "/usr/local/lib/python3.11/site-packages",
        "/usr/local/lib/python3.11/dist-packages",
        "/app/{}.runfiles/_main".format(binary_name),
    ]
    default_arg(kwargs, "env", {"PYTHONPATH": ":".join(python_path)})

    # Exclude rules_python dependencies (includes hermetic python toolchain and pip dependencies)
    kwargs["runfiles_denylist"].append("rules_python~")

    _gml_binary_image(
        name = name,
        binary = binary,
        **kwargs
    )

def _gml_minimal_py_image(name, binary, **kwargs):
    # We switch out the base image to include extra libs when we need to run a optimized ffmpeg.
    default_arg(kwargs, "base", select({
        "//bazel/cc_toolchains/sysroots:sysroot_type_cuda": "@gml//:cc_base_image",
        "//conditions:default": "@python_3_11_image",
    }))
    _gml_binary_image(
        name = name,
        binary = binary,
        **kwargs
    )

def gml_py_image(name, binary, **kwargs):
    default_arg(kwargs, "include_runfiles", True)

    default_arg(kwargs, "tags", [])
    if "manual" not in kwargs["tags"]:
        kwargs["tags"] = kwargs["tags"] + ["manual"]

    _gml_fast_py_image(name + ".fast", binary, **kwargs)
    _gml_minimal_py_image(name, binary, **kwargs)

gml_oci_image = _gml_oci_image
gml_binary_image = _gml_binary_image
gml_oci_push = _gml_oci_push
