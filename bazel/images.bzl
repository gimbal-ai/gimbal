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

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_image_index", "oci_tarball")
load("@rules_pkg//pkg:providers.bzl", "PackageFilesInfo")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel:lib.bzl", "default_arg")
load("//bazel:toolchain_transitions.bzl", "oci_image_arm64", "oci_image_x86_64")

def _gml_oci_image(name, multiarch = False, **kwargs):
    testonly = kwargs.pop("testonly", False)

    if "base" not in kwargs:
        default_arg(kwargs, "os", "linux")
        default_arg(kwargs, "architecture", "amd64")

    oci_image(
        name = name,
        testonly = testonly,
        **kwargs
    )

    oci_tarball(
        name = name + ".tar",
        image = ":" + name,
        # Workaround to match how Skaffold tags the image after building it.
        repo_tags = ["bazel/" + native.package_name() + ":" + name],
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
        **x86_kwargs
    )
    oci_image_arm64(
        name = name + "_arm64",
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
        _collect_runfiles(
            name = name + "_binary_runfiles",
            binaries = [binary],
            testonly = testonly,
        )
        tar_srcs.append(":" + name + "_binary_runfiles")

    pkg_tar(
        name = name + "_binary_tar",
        srcs = tar_srcs,
        package_dir = "/app",
        testonly = testonly,
    )

    kwargs["tars"] = kwargs["tars"] + [name + "_binary_tar"]
    _gml_oci_image(name, multiarch = multiarch, testonly = testonly, **kwargs)

def _collect_runfiles_impl(ctx):
    dest_to_src = dict()
    for binary in ctx.attr.binaries:
        binary_files = binary[DefaultInfo].files.to_list()
        if len(binary_files) != 1:
            fail("_collect_runfiles only works with binaries with a single file")
        binary_file = binary_files[0]
        binary_name = binary_file.basename
        runfiles_dir = binary_name + ".runfiles"

        runfiles = []
        if binary[DefaultInfo].default_runfiles:
            runfiles = binary[DefaultInfo].default_runfiles.files.to_list()
        for f in runfiles:
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
            path = "/".join([runfiles_dir, "_main", path])
            dest_to_src[path] = f

    return [
        PackageFilesInfo(
            dest_src_map = dest_to_src,
            attributes = {},
        ),
        DefaultInfo(
            files = depset(dest_to_src.values()),
        ),
    ]

_collect_runfiles = rule(
    implementation = _collect_runfiles_impl,
    attrs = dict(
        binaries = attr.label_list(),
    ),
)

def _jetson_host_ld_library_path():
    arch = "aarch64"
    paths = [
        "/lib/{arch}-linux-gnu",
        "/usr/lib/{arch}-linux-gnu",
        "/host_lib/{arch}-linux-gnu",
        "/host_cuda/lib64",
        "/host_lib/{arch}-linux-gnu/tegra",
    ]

    return ":".join(paths).format(arch = arch)

def _gml_jetson_image(name, binary, **kwargs):
    native.genrule(
        name = name + "_env_gen",
        outs = [name + ".env"],
        cmd = "> $@ echo LD_LIBRARY_PATH=" + _jetson_host_ld_library_path(),
    )

    default_arg(kwargs, "env", ":" + name + "_env_gen")
    _gml_binary_image(
        name,
        binary = binary,
        target_compatible_with = [
            "@platforms//cpu:aarch64",
        ] + select({
            "@gml//bazel/cc_toolchains:libc_version_glibc2_31": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        **kwargs
    )

gml_oci_image = _gml_oci_image
gml_binary_image = _gml_binary_image
gml_jetson_image = _gml_jetson_image
