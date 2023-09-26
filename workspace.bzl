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

# From tensorflow/workspace.bzl.
def _parse_bazel_version(bazel_version):
    # Remove commit from version.
    version = bazel_version.split(" ", 1)[0]

    # Split into (release, date) parts and only return the release
    # as a tuple of integers.
    parts = version.split("-", 1)

    # Turn "release" into a tuple of strings
    version_tuple = ()
    for number in parts[0].split("."):
        version_tuple += (str(number),)
    return version_tuple

# Check that a minimum version of bazel is being used.
def check_min_bazel_version(bazel_version):
    """Checks if bazel version is greater than passed in version

    Args:
        bazel_version: The bazel version to check against.
    """
    if "bazel_version" in dir(native) and native.bazel_version:
        current_bazel_version = _parse_bazel_version(native.bazel_version)
        minimum_bazel_version = _parse_bazel_version(bazel_version)
        if minimum_bazel_version > current_bazel_version:
            fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
                native.bazel_version,
                bazel_version,
            ))

# End: From tensorflow/workspace.bzl
