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

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_tarball")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel:lib.bzl", "default_arg")

def _gml_oci_image(name, **kwargs):
    oci_image(
        name = name,
        **kwargs
    )

    image_name = name
    if native.package_name():
        image_name = native.package_name() + "/" + image_name

    oci_tarball(
        name = name + ".load",
        image = ":" + name,
        repo_tags = [image_name + ":latest"],
        tags = ["manual"],
    )

def _gml_cc_image(name, binary, **kwargs):
    default_arg(kwargs, "base", "//:cc_base_image")
    default_arg(kwargs, "tars", [])
    default_arg(kwargs, "entrypoint", ["/app/" + Label(binary).name])

    pkg_tar(
        name = name + "_binary_tar",
        srcs = [binary],
        package_dir = "/app",
    )
    kwargs["tars"] = kwargs["tars"] + [name + "_binary_tar"]
    _gml_oci_image(name, **kwargs)

gml_oci_image = _gml_oci_image
gml_cc_image = _gml_cc_image
